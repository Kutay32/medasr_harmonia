import os
import csv
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.orm import declarative_base, sessionmaker

# Database configuration
DB_PATH = "sqlite:///data/patients.db"
engine = create_engine(DB_PATH, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PatientScan(Base):
    __tablename__ = "patient_scans"

    # Accession Number is unique per scan (order)
    accession_number = Column(String, primary_key=True, index=True)
    
    # Patient Demographics
    patient_id = Column(String, index=True)
    full_name = Column(String)
    gender = Column(String)
    dob = Column(Date)
    
    # Scan Details
    protocol_number = Column(String)
    scan_date = Column(Date)
    loinc_code = Column(String)
    clinic = Column(String)

def init_db():
    """Create tables if they don't exist."""
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)

def load_mock_data(csv_path="hastalar_mock.csv"):
    """Load mock data from CSV into the database."""
    if not os.path.exists(csv_path):
        print(f"Mock data file {csv_path} not found.")
        return

    db = SessionLocal()
    try:
        # Check if we already have data
        if db.query(PatientScan).first() is not None:
            print("Database already contains data. Skipping mock data load.")
            return

        with open(csv_path, mode='r', encoding='utf-8') as file:
            # Skip the first header line which has duplicates:
            # patient_id,patient_id,protocol_number,gender,scan_date,loinc_code,accession_number,dob,clinic
            next(file)
            
            # Read line by line
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    scan = PatientScan(
                        patient_id=parts[0],
                        full_name=parts[1],
                        protocol_number=parts[2],
                        gender=parts[3],
                        scan_date=datetime.strptime(parts[4], '%Y-%m-%d').date(),
                        loinc_code=parts[5],
                        accession_number=parts[6],
                        dob=datetime.strptime(parts[7], '%Y-%m-%d').date(),
                        clinic=parts[8]
                    )
                    db.add(scan)
            db.commit()
            print(f"Mock data loaded successfully from {csv_path}.")
            
    finally:
        db.close()

def get_patient_scans(patient_id=None):
    """Retrieve all scans for a patient, ordered by scan date (newest first)."""
    db = SessionLocal()
    try:
        query = db.query(PatientScan)
        if patient_id:
            query = query.filter(PatientScan.patient_id == patient_id)
        
        # Order by scan_date descending
        return query.order_by(PatientScan.scan_date.desc()).all()
    finally:
        db.close()

def get_scan_by_accession(accession_number):
    """Retrieve a specific scan by its accession number."""
    db = SessionLocal()
    try:
        return db.query(PatientScan).filter(PatientScan.accession_number == accession_number).first()
    finally:
        db.close()

def get_all_patients():
    """Retrieve a list of unique patients."""
    db = SessionLocal()
    try:
        # Get unique patient IDs and their names
        patients = db.query(PatientScan.patient_id, PatientScan.full_name).distinct().all()
        return [{"patient_id": p.patient_id, "full_name": p.full_name} for p in patients]
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    load_mock_data()
    print("Database initialized successfully.")
