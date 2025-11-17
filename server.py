# server.py
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Optional
import io
import sys
import time
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext
from PIL import Image  # (ImageStat no longer needed)
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.exc import IntegrityError

import numpy as np
import cv2

from ml.model_inference import (
    InferenceModel,
    resize_pad_square,
    _save_png_uint8,
    APP_ROOT,
)

# ---------- Paths / imports ----------
BASE_DIR = Path(__file__).resolve().parent

# Make local 'ml' package importable
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))



# ---------- App & CORS ----------
SECRET_KEY = "dev-secret-change-me"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8
DB_URL = "sqlite:///./app.db"
infer = InferenceModel() 
app = FastAPI(title="Dental Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you host frontend elsewhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Static dirs (for saved images) ----------
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
OUTPUTS_DIR = STATIC_DIR / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- Load trained model once ----------
MODEL_PATH = BASE_DIR / "ml" / "dental_unet_disease.keras"
LABELS_PATH = BASE_DIR / "ml" / "condition_labels.json"


# ---------- Home / assets ----------
@app.get("/", response_class=HTMLResponse)
def home():
    index = BASE_DIR / "ED2_TESTING.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("<h1>FastAPI is running</h1><p>Add ED2_TESTING.html next to server.py.</p>")
@app.get("/staff", response_class=HTMLResponse)
def staff_page():
    page = BASE_DIR / "staff.html"   # same folder as server.py
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="staff.html not found")

@app.get("/ED2_scc.css")
def css():
    return FileResponse(BASE_DIR / "ED2_scc.css")

@app.get("/JS_ED2.js")
def js():
    return FileResponse(BASE_DIR / "JS_ED2.js")

@app.get("/welcome", response_class=HTMLResponse)
def welcome():
    page = BASE_DIR / "welcome.html"
    if page.exists():
        return FileResponse(page)
    return HTMLResponse("""
    <!doctype html><meta charset="utf-8">
    <title>Welcome</title>
    <h1 id="greet">Welcome!</h1>
    <script>
      const name = sessionStorage.getItem('username') || 'User';
      document.getElementById('greet').textContent = `Welcome, ${name}!`;
    </script>
    """)

# ---------- DB ----------
Base = declarative_base()
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="patient")
    created_at = Column(DateTime, default=datetime.utcnow)
    patients = relationship("Patient", back_populates="owner")

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    name = Column(String, nullable=False)
    dob = Column(Date, nullable=True)
    sex = Column(String, nullable=True)
    note = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="patients")
    scans = relationship("Scan", back_populates="patient", cascade="all, delete-orphan")
    conditions = relationship("Condition", back_populates="patient", cascade="all, delete-orphan")

class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True, nullable=False)

    # paths like "/static/uploads/..." and "/static/outputs/..."
    masked_url = Column(String, nullable=False)
    overlay_url = Column(String, nullable=False)

    prediction = Column(String, nullable=True)
    confidence = Column(String, nullable=True)  # store as string "0.87" or similar

    note = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="scans")

class Condition(Base):
    __tablename__ = "conditions"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    name = Column(String, nullable=False)
    type = Column(String, nullable=True)         # e.g., "diagnosis", "physical", "dental"
    onset_date = Column(String, nullable=True)
    date_entered = Column(String, nullable=True) # store "YYYY-MM-DD"
    source = Column(String, nullable=True)
    entered_by = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="conditions")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Auth ----------
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto", bcrypt__truncate_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def hash_password(p: str) -> str: return pwd_context.hash(p)
def verify_password(p: str, h: str) -> bool: return pwd_context.verify(p, h)

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.password_hash):
        return None
    return user

def create_access_token(data: dict, minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + timedelta(minutes=minutes)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise cred_exc
    except JWTError:
        raise cred_exc
    user = get_user_by_username(db, username)
    if user is None:
        raise cred_exc
    return user

# ---------- Schemas ----------
class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "patient"
    name: str
    dob: Optional[date] = None
    sex: Optional[str] = None
    note: Optional[str] = None

class PatientIn(BaseModel):
    name: str
    dob: Optional[date] = None
    sex: Optional[str] = None
    note: Optional[str] = None

class PatientOut(BaseModel):
    id: int
    name: str
    dob: Optional[date] = None
    sex: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime
    class Config:
        from_attributes = True

class ScanOut(BaseModel):
    id: int
    masked_url: str
    overlay_url: str
    prediction: Optional[str] = None
    confidence: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ScanNoteUpdate(BaseModel):
    note: str

class StaffCreate(BaseModel):
    username: str
    password: str
class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "patient"
    name: str
    dob: Optional[date] = None
    sex: Optional[str] = None
    note: Optional[str] = None

class StaffCreate(BaseModel):
    username: str
    password: str

class ConditionCreate(BaseModel):
    name: str
    type: Optional[str] = None
    onset_date: Optional[str] = None
    source: Optional[str] = None


class ConditionOut(BaseModel):
    id: int
    name: str
    type: Optional[str]
    onset_date: Optional[str]
    date_entered: Optional[str]
    source: Optional[str]
    entered_by: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class PatientNoteUpdate(BaseModel):
    note: str

# ---------- Endpoints ----------
@app.post("/auth/register")
def register(u: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_username(db, u.username):
        raise HTTPException(status_code=400, detail="Username already exists.")
    try:
        user = User(username=u.username, password_hash=hash_password(u.password), role=u.role)
        db.add(user); db.commit(); db.refresh(user)
        if user.role == "patient":
            db.add(Patient(owner_id=user.id, name=u.name, dob=u.dob, sex=u.sex, note=u.note))
            db.commit()
        return {"message":"user created","user_id":user.id}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username already exists.")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

@app.post("/auth/token", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(400, "Incorrect username or password")
    token = create_access_token({"sub": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/patients", response_model=List[PatientOut])
def list_patients(current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current.role == "doctor":
        return db.query(Patient).order_by(Patient.created_at.desc()).all()
    return db.query(Patient).filter(Patient.owner_id == current.id).order_by(Patient.created_at.desc()).all()

@app.post("/api/patients", response_model=PatientOut)
def create_patient(p: PatientIn, current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patient = Patient(owner_id=current.id, name=p.name, dob=p.dob, sex=p.sex, note=p.note)
    db.add(patient); db.commit(); db.refresh(patient)
    return patient

def get_current_doctor(current: User = Depends(get_current_user)) -> User:
    if current.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Staff access only (doctor role required)",
        )
    return current


# ---------- Model-backed analyze endpoint ----------

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    # 1) Read uploaded bytes into OpenCV image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # 2) PRE-PROCESS exactly like Colab: pad/resize to 512×512
    masked_bgr = resize_pad_square(img_bgr, 512)

    # Save masked input (this is what you show on the LEFT)
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    masked_url = _save_png_uint8(masked_rgb, "masked")

    # 3) Run the SAME pipeline as in Colab using InferenceModel.save_overlay
    masked_path  = APP_ROOT / masked_url.lstrip("/")
    overlay_name = f"overlay_{int(time.time() * 1000)}.png"
    overlay_path = APP_ROOT / "static" / "outputs" / overlay_name

    overlay_bgr, legend = infer.save_overlay(str(masked_path), str(overlay_path))
    overlay_url = "/static/outputs/" + overlay_name

    # legend is like {"Cond 0": 63.8, "Cond 1": 49.2, ...}
    sorted_items = sorted(legend.items(), key=lambda x: x[1], reverse=True)
    if sorted_items:
        top_label, top_value = sorted_items[0]
        confidence = top_value / 100.0
    else:
        top_label, confidence = "condition", 0.0

    top3 = [
        {"label": name, "p": value / 100.0}
        for name, value in sorted_items[:3]
    ]

    return {
        "results": [
            {
                "masked_url": masked_url,     # LEFT: padded 512×512 X-ray
                "overlay_url": overlay_url,   # RIGHT: colored overlay + legend
                "prediction": top_label,
                "confidence": confidence,
                "top3": top3,
            }
        ]
    }


@app.post("/api/patients/{patient_id}/scans", response_model=ScanOut)
async def upload_and_analyze_scan(
    patient_id: int,
    file: UploadFile = File(...),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # --- check patient exists ---
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    # — doctor can access anyone; patients only themselves —
    if current.role != "doctor" and patient.owner_id != current.id:
        raise HTTPException(status_code=403, detail="Not allowed.")

    # 1) Read bytes into OpenCV image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # 2) Pre-process into padded 512x512
    masked_bgr = resize_pad_square(img_bgr, 512)
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

    masked_url = _save_png_uint8(masked_rgb, "masked")

    masked_path = APP_ROOT / masked_url.lstrip("/")
    overlay_name = f"overlay_{int(time.time() * 1000)}.png"
    overlay_path = APP_ROOT / "static" / "outputs" / overlay_name

    # 3) Run inference and overlay save
    overlay_bgr, legend = infer.save_overlay(str(masked_path), str(overlay_path))
    overlay_url = "/static/outputs/" + overlay_name

    # 4) Top prediction logic
    sorted_items = sorted(legend.items(), key=lambda x: x[1], reverse=True)
    if sorted_items:
        top_label, top_value = sorted_items[0]
        confidence = f"{top_value / 100.0:.3f}"   # store as string
    else:
        top_label, confidence = "condition", "0.0"

    # 5) Save scan to DB
    scan = Scan(
        patient_id=patient_id,
        masked_url=masked_url,
        overlay_url=overlay_url,
        prediction=top_label,
        confidence=confidence
    )

    db.add(scan)
    db.commit()
    db.refresh(scan)

    return scan

# ---------- CONDITION SCHEMAS ----------
class ConditionIn(BaseModel):
    name: str
    type: Optional[str] = None
    onset_date: Optional[str] = None
    source: Optional[str] = None

class ConditionOut(BaseModel):
    id: int
    name: str
    type: Optional[str]
    onset_date: Optional[str]
    date_entered: Optional[str]
    source: Optional[str]
    entered_by: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ---------- GET CONDITIONS FOR PATIENT ----------
@app.get("/api/patients/{patient_id}/conditions", response_model=List[ConditionOut])
def list_conditions(
    patient_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")

    if current.role != "doctor" and patient.owner_id != current.id:
        raise HTTPException(403, "Not allowed")

    return db.query(Condition)\
        .filter(Condition.patient_id == patient_id)\
        .order_by(Condition.created_at.desc())\
        .all()


# ---------- CREATE CONDITION ----------
@app.post("/api/patients/{patient_id}/conditions", response_model=ConditionOut)
def create_condition(
    patient_id: int,
    data: ConditionIn,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")

    if current.role != "doctor":
        raise HTTPException(403, "Only doctors can add conditions")

    condition = Condition(
        patient_id=patient_id,
        name=data.name,
        type=data.type,
        onset_date=data.onset_date,
        date_entered=datetime.utcnow().strftime("%Y-%m-%d"),
        source=data.source,
        entered_by=current.username
    )

    db.add(condition)
    db.commit()
    db.refresh(condition)
    return condition


@app.get("/api/patients/{patient_id}/scans", response_model=List[ScanOut])
def list_scans_for_patient(
    patient_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    if current.role != "doctor" and patient.owner_id != current.id:
        raise HTTPException(status_code=403, detail="Not allowed for this patient.")

    scans = (
        db.query(Scan)
        .filter(Scan.patient_id == patient_id)
        .order_by(Scan.created_at.desc())
        .all()
    )
    return scans



@app.get("/api/me")
def me(current: User = Depends(get_current_user)):
    return {
        "username": current.username,
        "role": current.role,
        "created_at": current.created_at.isoformat(),
    }

@app.post("/api/staff/create_member")
def create_staff_member(
    data: StaffCreate,
    current: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    # Check if username already exists
    if get_user_by_username(db, data.username):
        raise HTTPException(status_code=400, detail="Username already exists.")

    user = User(
        username=data.username,
        password_hash=hash_password(data.password),
        role="doctor", 
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "message": "Staff account created",
        "username": user.username,
        "role": user.role,
        "created_at": user.created_at.isoformat(),
    }

@app.patch("/api/scans/{scan_id}/note", response_model=ScanOut)
def update_scan_note_endpoint(
    scan_id: int,
    body: ScanNoteUpdate,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    patient = db.query(Patient).filter(Patient.id == scan.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    if current.role != "doctor" and patient.owner_id != current.id:
        raise HTTPException(status_code=403, detail="Not allowed for this patient.")

    scan.note = body.note
    db.commit()
    db.refresh(scan)
    return scan

@app.patch("/api/patients/{patient_id}/note", response_model=PatientOut)
def update_patient_note(
    patient_id: int,
    body: PatientNoteUpdate,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    # doctor can edit any; patient can only edit their own note
    if current.role != "doctor" and patient.owner_id != current.id:
        raise HTTPException(status_code=403, detail="Not allowed.")

    patient.note = body.note
    db.commit()
    db.refresh(patient)
    return patient
