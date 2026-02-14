"""
Modelos de base de datos usando SQLAlchemy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Person(Base):
    """Modelo de persona"""
    __tablename__ = 'persons'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    reports = relationship("Report", back_populates="person")

    def to_dict(self):
        """Convierte el modelo a diccionario"""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'deleted_at': self.deleted_at.isoformat() if self.deleted_at else None,
        }


class Report(Base):
    """Modelo de reporte"""
    __tablename__ = 'reports'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=False, index=True)
    observations = Column(Text, nullable=False)
    evidence = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    person = relationship("Person", back_populates="reports")

    def to_dict(self):
        """Convierte el modelo a diccionario"""
        return {
            'id': self.id,
            'person_id': self.person_id,
            'observations': self.observations,
            'evidence': self.evidence,
            'created_at': self.created_at.isoformat(),
            'deleted_at': self.deleted_at.isoformat() if self.deleted_at else None,
        }
