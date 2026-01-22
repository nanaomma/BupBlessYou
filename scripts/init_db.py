"""Database initialization script"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.connection import Base, engine
from src.database.models import Case, Conversation, Judgment


def init_database():
    """Initialize database tables"""
    print("Creating database tables...")

    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")

        # List created tables
        print("\nCreated tables:")
        for table in Base.metadata.sorted_tables:
            print(f"  - {table.name}")

    except Exception as e:
        print(f"Error creating tables: {e}")
        raise


def drop_all_tables():
    """Drop all database tables (use with caution!)"""
    confirm = input("⚠️ This will delete all data. Are you sure? (yes/no): ")

    if confirm.lower() == "yes":
        print("Dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        print("All tables dropped!")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database initialization")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all tables (dangerous!)"
    )

    args = parser.parse_args()

    if args.drop:
        drop_all_tables()
    else:
        init_database()
