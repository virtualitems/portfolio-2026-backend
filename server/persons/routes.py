"""
CRUD endpoints para la gestión de personas.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ..shared.database import get_db
from ..shared.models import Person
from ..shared.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix='/persons', tags=['persons'])


class PersonCreate(BaseModel):
    name: str
    email: EmailStr


class PersonUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None


class PersonResponseModel(BaseModel):
    id: int
    name: str
    email: str
    created_at: str


class SinglePersonResponse(BaseModel):
    data: PersonResponseModel


class ListPersonResponse(BaseModel):
    data: List[PersonResponseModel]


@router.get('', response_model=ListPersonResponse)
async def get_persons(db: Session = Depends(get_db)):
    """
    Obtiene todas las personas que no han sido eliminadas.
    """
    try:
        persons = db.query(Person).filter(Person.deleted_at.is_(None)).all()
        return {"data": [person.to_dict() for person in persons]}
    except Exception as e:
        logger.error(f"Error getting persons: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{person_id}', response_model=SinglePersonResponse)
async def get_person(person_id: int, db: Session = Depends(get_db)):
    """
    Obtiene una persona por su ID.
    """
    try:
        person = db.query(Person).filter(
            Person.id == person_id,
            Person.deleted_at.is_(None)
        ).first()

        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        return {"data": person.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting person {person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post('', status_code=204)
async def create_person(person: PersonCreate, db: Session = Depends(get_db)):
    """
    Crea una nueva persona.
    """
    try:
        db_person = Person(
            name=person.name,
            email=person.email
        )
        db.add(db_person)
        db.commit()

        return None
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Error creating person: {e}")
        raise HTTPException(status_code=400, detail="Email already exists")
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put('/{person_id}', status_code=204)
async def update_person(person_id: int, person: PersonUpdate, db: Session = Depends(get_db)):
    """
    Actualiza una persona existente.
    """
    try:
        db_person = db.query(Person).filter(
            Person.id == person_id,
            Person.deleted_at.is_(None)
        ).first()

        if not db_person:
            raise HTTPException(status_code=404, detail="Person not found")

        if person.name is None and person.email is None:
            raise HTTPException(status_code=400, detail="No fields to update")

        if person.name is not None:
            db_person.name = person.name

        if person.email is not None:
            db_person.email = person.email

        db_person.updated_at = datetime.utcnow()
        db.commit()

        return None
    except HTTPException:
        raise
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Error updating person {person_id}: {e}")
        raise HTTPException(status_code=400, detail="Email already exists")
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating person {person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/{person_id}', status_code=204)
async def delete_person(person_id: int, db: Session = Depends(get_db)):
    """
    Realiza un borrado lógico de una persona (soft delete).
    """
    try:
        db_person = db.query(Person).filter(
            Person.id == person_id,
            Person.deleted_at.is_(None)
        ).first()

        if not db_person:
            raise HTTPException(status_code=404, detail="Person not found")

        db_person.deleted_at = datetime.utcnow()
        db.commit()

        return None
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting person {person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
