"""
CRUD endpoints para la gestión de reportes.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Form, UploadFile, File, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..shared.database import get_db
from ..shared.models import Report, Person
from ..shared.files import save_image_file, delete_file
from ..shared.logger import get_logger
from ..shared.env import env

logger = get_logger(__name__)

router = APIRouter(prefix='/reports', tags=['reports'])


class PersonEmbeddedModel(BaseModel):
    name: str
    email: str


class ReportResponseModel(BaseModel):
    id: int
    person: PersonEmbeddedModel
    evidence: str
    observations: str
    created_at: str


class SingleReportResponse(BaseModel):
    data: ReportResponseModel


class ListReportResponse(BaseModel):
    data: List[ReportResponseModel]

@router.get('', response_model=ListReportResponse)
async def get_reports(person_id: Optional[int] = None, db: Session = Depends(get_db)):
    """
    Obtiene todos los reportes que no han sido eliminados.
    Opcionalmente filtra por person_id.
    Incluye información de la persona asociada.
    """
    try:
        query = db.query(Report, Person).join(
            Person, Report.person_id == Person.id
        ).filter(
            Report.deleted_at.is_(None),
            Person.deleted_at.is_(None)
        )

        if person_id:
            query = query.filter(Report.person_id == person_id)

        results = query.all()

        base_url = env.get('BASE_URL', '').rstrip('/')

        reports_data = [
            {
                'id': report.id,
                'person': {
                    'name': person.name,
                    'email': person.email
                },
                'evidence': f"{base_url}/media/{report.evidence}",
                'observations': report.observations,
                'created_at': report.created_at.isoformat() if report.created_at else None
            }
            for report, person in results
        ]

        return {"data": reports_data}
    except Exception as e:
        logger.error(f"Error getting reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{report_id}', response_model=SingleReportResponse)
async def get_report(report_id: int, db: Session = Depends(get_db)):
    """
    Obtiene un reporte por su ID.
    Incluye información de la persona asociada.
    """
    try:
        result = db.query(Report, Person).join(
            Person, Report.person_id == Person.id
        ).filter(
            Report.id == report_id,
            Report.deleted_at.is_(None),
            Person.deleted_at.is_(None)
        ).first()

        if not result:
            raise HTTPException(status_code=404, detail="Report not found")

        report, person = result
        base_url = env.get('BASE_URL', '').rstrip('/')
        report_data = {
            'id': report.id,
            'person': {
                'name': person.name,
                'email': person.email
            },
            'evidence': f"{base_url}/media/{report.evidence}",
            'observations': report.observations,
            'created_at': report.created_at.isoformat() if report.created_at else None
        }

        return {"data": report_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post('', status_code=204)
async def create_report(
    person_id: int = Form(...),
    observations: str = Form(""),
    evidence: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Crea un nuevo reporte.
    El campo evidence debe ser un archivo de imagen (multipart/form-data).
    Las observations son opcionales (por defecto string vacío).
    """
    try:

        if not evidence:
            raise HTTPException(status_code=400, detail="Evidence is required")

        person = db.query(Person).filter(
            Person.id == person_id,
            Person.deleted_at.is_(None)
        ).first()

        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        try:
            image_data = await evidence.read()

            if not image_data or len(image_data) == 0:
                raise HTTPException(status_code=400, detail="Evidence cannot be blank")

            extension = evidence.content_type.split('/')[-1]

            evidence_path = save_image_file(
                image_data,
                storage_dir='mediafiles',
                prefix='evidence',
                extension=extension
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")

        db_report = Report(
            person_id=person_id,
            observations=observations,
            evidence=evidence_path
        )
        db.add(db_report)
        db.commit()

        return None
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/{report_id}', status_code=204)
async def delete_report(report_id: int, db: Session = Depends(get_db)):
    """
    Realiza un borrado lógico de un reporte (soft delete).
    Opcionalmente elimina el archivo de evidencia.
    """
    try:
        db_report = db.query(Report).filter(
            Report.id == report_id,
            Report.deleted_at.is_(None)
        ).first()

        if not db_report:
            raise HTTPException(status_code=404, detail="Report not found")

        db_report.deleted_at = datetime.utcnow()
        db.commit()

        if db_report.evidence:
            try:
                delete_file(db_report.evidence, 'mediafiles')
            except Exception as e:
                logger.warning(f"Could not delete evidence file: {e}")

        return None
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
