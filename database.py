from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import DateTime
import pyotp

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False, default='user')
    email = Column(String, nullable=False) 
    totp_secret = Column(String)
    reset_token = Column(String) 
    reset_token_expiry = Column(DateTime) 



engine = create_engine('sqlite:///users.db')  # SQLite database
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def seed_users():
    session = Session()
    # Check if users already exist
    if session.query(User).count() == 0:
        users = [
                User(
                    username="admin",
                    password="admin123",  # Make sure this meets password requirements
                    role="admin",
                    email="admin@example.com",
                    totp_secret=generate_totp_secret(),
                    reset_token=None,
                    reset_token_expiry=None
                ),
                User(
                    username="user1",
                    password="password123",  # Make sure this meets password requirements
                    role="user",
                    email="user1@example.com",
                    totp_secret=generate_totp_secret(),
                    reset_token=None,
                    reset_token_expiry=None
                ),
            ]
        session.add_all(users)
        session.commit()
    session.close()

def generate_totp_secret():
    return pyotp.random_base32()
