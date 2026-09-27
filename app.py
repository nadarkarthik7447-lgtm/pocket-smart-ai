from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
import os
import json
import re
import base64
import shutil
import urllib.parse
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import uuid
import asyncio

from fastapi import (
    FastAPI, HTTPException, Depends, File, UploadFile,
    Form, Request, status, Cookie
)
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

import gemini_utils

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI(title="PocketSmart: AI Budget Planner")

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

os.makedirs(os.path.join(static_dir, "uploads"), exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# In-memory stores
users_db: Dict[str, dict] = {}
active_sessions: Dict[str, Any] = {}
user_recommendations: Dict[str, List[Any]] = {}
blacklisted_tokens = set()

# Pydantic Schemas
class RegisterUser(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

    class Config:
        extra = "allow"
    confirm_password: Optional[str] = None

    class Config:
        extra = "ignore"
class Token(BaseModel):
    access_token: str
    token_type: str

class UserInDB(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None

class HomeBudgetInput(BaseModel):
    total_budget: float
    num_lights: int = 0
    num_fans: int = 0
    num_furniture: int = 0
    num_dining_tables: int = 0
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = None

class PartyBudgetInput(BaseModel):
    total_budget: float
    party_type: str
    num_guests: int
    venue_type: Optional[str] = "Not specified"
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = True
    additional_requirements: Optional[str] = None

class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str
    preferences: Optional[str] = None

class UserSession:
    def __init__(self, username: str, token: str, user_data: dict = None):
        self.username = username
        self.token = token
        self.login_time = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.user_data = user_data or {}

class RecommendationItem:
    def __init__(self, rec_id: str, rec_type: str, input_summary: dict, result: dict):
        self.id = rec_id
        self.timestamp = datetime.utcnow().strftime("%B %d, %Y, %I:%M %p")
        self.recommendation_type = rec_type
        self.input_summary = input_summary
        self.full_result = result
        self.result_summary = {
            "total_budget": result.get("total_budget", 0),
            "remaining_budget": result.get("remaining_budget", 0)
        }

# Helpers
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def authenticate_user(db: dict, username: str, password: str):
    user = db.get(username)
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return UserInDB(username=user["username"], email=user["email"], full_name=user.get("full_name"))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token(request: Request) -> Optional[str]:
    # Check Cookie first
    token = request.cookies.get("access_token")
    if token:
        return token
    # Fallback to Authorization Header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    return None

async def get_current_user(request: Request, token: Optional[str] = Depends(get_token)) -> UserInDB:
    if not token or token in blacklisted_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        
        if username in active_sessions:
            active_sessions[username].last_activity = datetime.utcnow()

        user = users_db[username]
        return UserInDB(username=user["username"], email=user["email"], full_name=user.get("full_name"))
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    return current_user

def save_upload_file(image: UploadFile) -> str:
    filename = f"{uuid.uuid4()}_{image.filename}"
    file_path = os.path.join("static/uploads", filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    return file_path

def save_to_history(username: str, recommendation_type: str, input_data: dict, result: dict):
    if username not in user_recommendations:
        user_recommendations[username] = []
    rec = RecommendationItem(
        rec_id=str(uuid.uuid4()),
        rec_type=recommendation_type,
        input_summary=input_data,
        result=result
    )
    user_recommendations[username].append(rec)
    return rec

def usd_to_inr(amount_usd: float, exchange_rate: float = 83.0) -> float:
    """Convert USD amount to INR using the specified exchange rate"""
    return amount_usd * exchange_rate

# Background task to clean up expired sessions
@app.on_event("startup")
async def setup_session_cleanup():
    """Background task to clean up expired sessions"""
    async def cleanup_expired_sessions():
        while True:
            current_time = datetime.utcnow()
            # Check for sessions that have been inactive for more than 30 minutes
            expired_sessions = [
                username for username, session in active_sessions.items()
                if (current_time - session.last_activity).total_seconds() > 1800  # 30 minutes
            ]
            
            # Remove expired sessions
            for username in expired_sessions:
                if username in active_sessions:
                    print(f"Removing expired session for {username}")
                    del active_sessions[username]
                    
            # Wait for 5 minutes before checking again
            await asyncio.sleep(300)

    # Start the background task
    asyncio.create_task(cleanup_expired_sessions())

# ---------------- Routes: HTML Views ----------------

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    with open("templates/login.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
    

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    with open("templates/register.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    history = user_recommendations.get(current_user.username, [])
    sorted_history = sorted(history, key=lambda x: x.timestamp, reverse=True)[:5]
    return templates.TemplateResponse(
        request=request, 
        name="dashboard.html", 
        context={"user": current_user, "recent_activity": sorted_history}
    )

@app.get("/home-planner")
async def home_planner():
    file_path = os.path.join(templates_dir, "home_planner.html")
    return FileResponse(file_path)

@app.get("/party-planner")
async def party_planner():
    file_path = os.path.join(templates_dir, "party_planner.html")
    return FileResponse(file_path)

@app.get("/jewelry-planner")
async def jewelry_planner():
    file_path = os.path.join(templates_dir, "jewelry_planner.html")
    return FileResponse(file_path)

@app.get("/testimonials")
async def get_testimonials():
    file_path = os.path.join(templates_dir, "testimonials.html")
    return FileResponse(file_path)

@app.get("/history")
async def history_page():
    file_path = os.path.join(templates_dir, "history.html")
    return FileResponse(file_path)
# ---------------- Authentication Endpoints ----------------

@app.post("/register")
@app.post("/api/auth/register")
async def register(user_data: RegisterUser):
    if user_data.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    users_db[user_data.username] = {
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name or user_data.username,
        "password": get_password_hash(user_data.password)
    }
    return {"message": "User registered successfully"}

@app.post("/token", response_model=Token)
@app.post("/api/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint to get access token"""
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Create or update session for the user
    existing_user_data = {}
    if user.username in active_sessions:
        existing_user_data = active_sessions[user.username].user_data
        # If there's an old token, blacklist it
        old_token = active_sessions[user.username].token
        blacklisted_tokens.add(old_token)
    
    active_sessions[user.username] = UserSession(
        username=user.username,
        token=access_token,
        user_data=existing_user_data
    )
    
    # Return response with cookie
    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    
    response.set_cookie(
        key="access_token",
        value=access_token, # Store token directly without Bearer prefix
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax"
    )
    
    return response

@app.post("/logout")
async def logout(request: Request):
    """Logout user by blacklisting their token and clearing session"""
    token = await get_token(request)
    
    if token:
        # Add token to blacklist
        blacklisted_tokens.add(token)
        
        try:
            # Extract username from token to remove from active sessions
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username and username in active_sessions:
                del active_sessions[username]
        except JWTError:
            pass
            
    # Use RedirectResponse instead of JSONResponse for proper redirection
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response

# ---------------- Session Info Endpoints ----------------

@app.get("/session-info")
async def get_session_info(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """Get current user's session information"""
    if current_user.username in active_sessions:
        session = active_sessions[current_user.username]
        return {
            "username": session.username,
            "login_time": session.login_time,
            "last_activity": session.last_activity,
            "session_duration": (datetime.utcnow() - session.login_time).total_seconds() // 60, # in minutes
            "user_data": session.user_data
        }
    else:
        raise HTTPException(status_code=404, detail="No active session found")

@app.post("/session-data")
async def update_session_data(
    data: Dict[str, Any],
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Update user's session data"""
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data.update(data)
        active_sessions[current_user.username].last_activity = datetime.utcnow()
        return {"message": "Session data updated", "data": active_sessions[current_user.username].user_data}
    else:
        raise HTTPException(status_code=404, detail="No active session found")

# ---------------- Budget Planner Routes ----------------

@app.post("/home-budget")
async def plan_home_budget(
    budget_input: HomeBudgetInput,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Generate home budget recommendations"""
    # Store last budget planning in session data
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_home_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "requirements": {
                "lights": budget_input.num_lights,
                "fans": budget_input.num_fans,
                "furniture": budget_input.num_furniture,
                "dining_tables": budget_input.num_dining_tables
            }
        }
    
    # Get recommendations
    result = gemini_utils.get_home_recommendations(budget_input)
    
    # Save to history
    save_to_history(
        username=current_user.username,
        recommendation_type="home",
        input_data=budget_input.dict(),
        result=result
    )
    
    return result

@app.post("/party-budget")
async def plan_party_budget(
    budget_input: PartyBudgetInput,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Generate party budget recommendations"""
    # Store last budget planning in session data
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_party_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "party_type": budget_input.party_type,
            "guests": budget_input.num_guests
        }
    
    # Get recommendations
    result = gemini_utils.get_party_recommendations(budget_input)
    
    # Save to history
    save_to_history(
        username=current_user.username,
        recommendation_type="party",
        input_data=budget_input.dict(),
        result=result
    )
    
    return result

@app.post("/jewelry-budget")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    request: Request = None,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Generate jewelry budget recommendations with optional outfit image"""
    # Create the budget input model
    budget_input = JewelryBudgetInput(
        total_budget=total_budget,
        occasion=occasion,
        preferences=preferences
    )
    
    image_path = None
    if image and image.filename:
        image_path = save_upload_file(image)
        
    # Store last budget planning in session data
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_jewelry_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "occasion": budget_input.occasion,
            "has_image": image is not None
        }
        
    # Get recommendations
    result = gemini_utils.get_jewelry_recommendations(budget_input, image_path)
    
    # Save to history with image info
    input_data = budget_input.dict()
    if image and image.filename:
        input_data["image"] = image.filename
        
    save_to_history(
        username=current_user.username,
        recommendation_type="jewelry",
        input_data=input_data,
        result=result
    )
    
    return result

# ---------------- History & Details Endpoints ----------------

@app.get("/recommendation-history")
async def get_recommendation_history(
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Get the user's recommendation history"""
    if current_user.username not in user_recommendations:
        return {"history": []}
        
    # Sort history by timestamp (newest first)
    history = sorted(
        user_recommendations[current_user.username],
        key=lambda x: x.timestamp,
        reverse=True
    )
    
    # Convert to dict for JSON response
    history_data = []
    for item in history:
        history_data.append({
            "id": item.id,
            "timestamp": item.timestamp,
            "type": item.recommendation_type,
            "input": item.input_summary,
            "summary": item.result_summary
        })
        
    return {"history": history_data}

@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(
    recommendation_id: str,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Get the full details of a specific recommendation"""
    if current_user.username not in user_recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found")
        
    # Find the recommendation with the given ID
    for item in user_recommendations[current_user.username]:
        if item.id == recommendation_id:
            return {
                "id": item.id,
                "timestamp": item.timestamp,
                "type": item.recommendation_type,
                "input": item.input_summary,
                "full_result": item.full_result
            }
            
    raise HTTPException(status_code=404, detail="Recommendation not found")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    print("Starting PocketSmart: AI Budget Planner...")
    uvicorn.run(app, host="0.0.0.0", port=8000)