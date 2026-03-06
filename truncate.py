#!/usr/bin/env python3
"""
Script para hacer truncate a todas las tablas en Supabase.
Uso: python truncate.py
"""
import sys
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()


def get_connection():
    """Crea y retorna una conexión a la base de datos de Supabase."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        return conn
    except Exception as e:
        print(f"❌ Error al conectar a la base de datos: {e}")
        sys.exit(1)


def get_all_tables(cursor):
    """Obtiene todas las tablas de la base de datos (excluyendo tablas del sistema)."""
    cursor.execute("""
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename;
    """)
    return [row[0] for row in cursor.fetchall()]


def truncate_all_tables(conn):
    """Hace truncate a todas las tablas en la base de datos."""
    cursor = conn.cursor()

    try:
        # Obtener lista de tablas
        tables = get_all_tables(cursor)

        if not tables:
            print("⚠️  No se encontraron tablas en la base de datos.")
            return

        print(f"📋 Se encontraron {len(tables)} tabla(s):")
        for table in tables:
            print(f"   - {table}")

        # Confirmar acción
        print("\n⚠️  ADVERTENCIA: Esta acción eliminará TODOS los datos de las tablas.")
        confirm = input("¿Estás seguro de que deseas continuar? (escribe 's/S' para confirmar): ")

        if confirm != 's' and confirm != 'S':
            print("❌ Operación cancelada.")
            return

        print("\n🔄 Iniciando truncate...")

        # Deshabilitar temporalmente las restricciones de clave foránea
        cursor.execute("SET session_replication_role = 'replica';")

        # Hacer truncate a cada tabla
        for table in tables:
            try:
                cursor.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE;").format(
                    sql.Identifier(table)
                ))
                print(f"✅ Truncate exitoso en tabla: {table}")
            except Exception as e:
                print(f"❌ Error al hacer truncate en tabla {table}: {e}")

        # Rehabilitar las restricciones de clave foránea
        cursor.execute("SET session_replication_role = 'origin';")

        # Confirmar los cambios
        conn.commit()
        print("\n✅ Truncate completado exitosamente en todas las tablas.")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error durante el truncate: {e}")
        sys.exit(1)
    finally:
        cursor.close()


def main():
    """Función principal del script."""
    print("=" * 60)
    print("🗑️  SCRIPT DE TRUNCATE PARA SUPABASE")
    print("=" * 60)

    # Conectar a la base de datos
    conn = get_connection()
    print("✅ Conexión exitosa a la base de datos.\n")

    try:
        # Ejecutar truncate
        truncate_all_tables(conn)
    finally:
        # Cerrar conexión
        conn.close()
        print("\n🔌 Conexión cerrada.")


if __name__ == "__main__":
    main()
