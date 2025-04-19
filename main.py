import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv
import easyocr
from PIL import Image
import stripe  # Stripe integration
from motor.motor_asyncio import AsyncIOMotorClient  # Async MongoDB client
import certifi
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional  # Added to allow Optional types
from fastapi import Query

# Load environment variables from .env file
load_dotenv()

# ---------------------------
# APP INITIALIZATION & MIDDLEWARE
# ---------------------------
app = FastAPI(title="Likemeornot AI Analysis API")
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# MONGODB SETUP (using Motor)
# ---------------------------
MONGO_DETAILS = os.getenv("MONGO_DETAILS")
client = AsyncIOMotorClient(
     MONGO_DETAILS,
     tls=True,
     tlsCAFile=certifi.where()      # <— this points Motor at a known CA bundle
 )
database = client['LikeMeOrNot']  # Replace with your actual database name
user_collection = database.get_collection("users")  # Collection for user authentication

# ---------------------------
# AUTHENTICATION SETUP (JWT)
# ---------------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    print("DEBUG: Created JWT with payload:", to_encode)  # Debug log
    return encoded_jwt



async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        # Retrieve user details from MongoDB
        user = await user_collection.find_one({"username": username})
        print("DEBUG: Retrieved user from DB:", user)
        if not user:
            # If the user is not found, automatically create a new user with default values.
            new_user = {
                "username": username,
                "email": "",
                "password": "",
                "subscription": "free",
                "subscription_expiry": None
            }
            result = await user_collection.insert_one(new_user)
            user = new_user
            user["id"] = str(result.inserted_id)
            print("DEBUG: New user auto-created:", user)
        return {"username": username, "subscription": user.get("subscription", "free")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

# Token endpoint for user login (for simplicity, no password verification is implemented here).
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    access_token = create_access_token(data={"sub": form_data.username})
    print("DEBUG: Login endpoint returning token for user:", form_data.username)
    return {"access_token": access_token, "token_type": "bearer"}

# ---------------------------
# SETUP OPENAI & STRIPE
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
print("Loaded OpenAI API Key:", bool(openai.api_key))

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
print("Loaded Stripe API Key:", bool(stripe.api_key))

# ---------------------------
# MODELS FOR AI ANALYSIS & USERS
# ---------------------------
class AnalysisRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    report: str

# Update the User model to allow subscription_expiry to be optional.
class User(BaseModel):
    username: str = Field(...)
    email: str = Field(...)
    password: str = Field(...)  # In production, store hashed passwords.
    subscription: str = Field(default="free")  # "free", "one_time", or "monthly"
    subscription_expiry: Optional[datetime] = None  # Allow None for no expiry

class UserResponse(User):
    id: str

@app.get("/confirm-subscription")
async def confirm_subscription(
    session_id: str = Query(...),
    current_user: dict = Depends(get_current_user)
):
    # retrieve the session directly from Stripe:
    session = stripe.checkout.Session.retrieve(session_id)
    mode = session.mode  # "subscription" or "payment"
    if mode == "subscription":
        expiry   = datetime.utcnow() + timedelta(days=30)
        sub_type = "monthly"
    else:
        expiry   = datetime.utcnow()
        sub_type = "one_time"
    await user_collection.update_one(
        {"username": current_user["username"]},
        {"$set": {
            "subscription": sub_type,
            "subscription_expiry": expiry
        }}
    )
    return {"status": "ok", "subscription": sub_type, "expiry": expiry}

@app.get(
  "/users/me", 
  response_model=UserResponse,
  summary="Return the currently‐logged‐in user record"
)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """
    - `get_current_user` decodes the JWT and gives us `{"username": ...}`.
    - Here we load the full MongoDB record (so we get email, subscription, expiry).
    """
    db_user = await user_collection.find_one({"username": current_user["username"]})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    # convert Mongo _id to str id and match our Pydantic model
    db_user["id"] = str(db_user["_id"])
    return UserResponse(**db_user)
# ---------------------------
# USER MANAGEMENT ENDPOINTS (Optional)
# ---------------------------
@app.post("/users", response_model=UserResponse)
async def create_user(user: User):
    try:
        user_dict = user.dict()
        result = await user_collection.insert_one(user_dict)
        user_response = {**user_dict, "id": str(result.inserted_id)}
        print("DEBUG: Created user:", user_response)
        return user_response
    except Exception as e:
        # print the full stack in your console so you can see exactly what blew up
        import traceback; traceback.print_exc()
        # return a clean error to the client (with CORS headers)
        raise HTTPException(status_code=500, detail=f"Could not create user: {e}")



@app.get("/users/{username}", response_model=UserResponse)
async def get_user(username: str):
    user = await user_collection.find_one({"username": username})
    if user:
        user["id"] = str(user["_id"])
        print("DEBUG: Retrieved user:", user)
        return user
    raise HTTPException(status_code=404, detail="User not found")


@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """
    Returns the full user record (including subscription & expiry)
    for whoever’s JWT they sent.
    """
    db_user = await user_collection.find_one({"username": current_user["username"]})
    if not db_user:
        raise HTTPException(404, "User not found")
    db_user["id"] = str(db_user["_id"])
    return db_user

# ---------------------------
# HELPER FUNCTION: Determine Subscription Expiry
# ---------------------------
def determine_subscription_expiry(subscription_type: str) -> Optional[datetime]:
    now = datetime.utcnow()
    if subscription_type == "monthly":
        # For a monthly subscription, expire in 30 days.
        return now + timedelta(days=30)
    elif subscription_type == "one_time":
        # For one-time use, expire immediately after use.
        return now
    else:
        return None

# ---------------------------
# UPDATED TEXT ANALYSIS ENDPOINT (requires login)
# ---------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest, current_user: dict = Depends(get_current_user)):
    prompt = (
        "Analyze the following conversation and return a JSON object with the following keys exactly:\n"
        '{\n  "score": <string representing the likability score>,\n'
        '  "tone": <string describing the tone and emotions>,\n'
        '  "keyPhrases": [<array of key phrases>],\n'
        '  "tips": <string with a brief advice for the next reply>\n'
        '}\n\n'
        "Conversation:\n"
        f"{request.text}\n\n"
        "Please return only valid JSON."
    )
    try:
        print("Prompt being sent to OpenAI:", prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        report = response.choices[0].message.content.strip()
        print("Full OpenAI response:", response)
    except Exception as e:
        report = "Error analyzing text."
        print("OpenAI API error:", e)
    return AnalysisResponse(report=report)

# ---------------------------
# UPDATED IMAGE ANALYSIS ENDPOINT (requires login)
# ---------------------------
@app.post("/analyze-image", response_model=AnalysisResponse)
async def analyze_image(image: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    print("Endpoint /analyze-image triggered")
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)
        print("Converted image to NumPy array, shape:", img_np.shape)
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(img_np, detail=0)
        extracted_text = " ".join(results)
        print("Extracted text from image:", extracted_text)
        prompt = (
            "Analyze the following conversation extracted from an image and return a JSON object with the following keys exactly:\n"
            '{\n  "score": <string representing the likability score>,\n'
            '  "tone": <string describing the tone and emotions>,\n'
            '  "keyPhrases": [<array of key phrases>],\n'
            '  "tips": <string with a brief advice for the next reply>\n'
            '}\n\n'
            "Conversation:" + extracted_text + "\n\nPlease return only valid JSON."
        )
        print("Prompt for image analysis:", prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        report = response.choices[0].message.content.strip()
    except Exception as e:
        report = f"Error analyzing image: {e}"
        print("Error analyzing image:", e)
    return AnalysisResponse(report=report)

# ---------------------------
# STRIPE CHECKOUT ENDPOINT (ONE-TIME PAYMENT)
# ---------------------------
class CheckoutSessionResponse(BaseModel):
    sessionId: str

@app.post("/create-checkout-session", response_model=CheckoutSessionResponse)
async def create_checkout_session(current_user: dict = Depends(get_current_user)):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "gbp",
                        "product_data": {"name": "Unlock My Next Move"},
                        "unit_amount": 99,
                    },
                    "quantity": 1,
                },
            ],
            mode="payment",
            success_url="http://localhost:3000/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="http://localhost:3000/cancel",
            metadata={"username": current_user["username"]}
        )
        print("Stripe Checkout Session created:", session)
        return CheckoutSessionResponse(sessionId=session.id)
    except Exception as e:
        print("Error creating checkout session:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# STRIPE SUBSCRIPTION ENDPOINT
# ---------------------------
STRIPE_SUB_PRICE = os.getenv("STRIPE_SUB_PRICE") 

@app.post("/create-subscription-session", response_model=CheckoutSessionResponse)
async def create_subscription_session(current_user: dict = Depends(get_current_user)):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {"price": "price_1REDAYLIfxI8v49W6dzoFRZ0", "quantity": 1},
            ],
            mode="subscription",
            success_url="http://localhost:3000/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="http://localhost:3000/cancel",
            metadata={"username": current_user["username"]}
        )
        print("Stripe Subscription Session created:", session)
        return CheckoutSessionResponse(sessionId=session.id)
    except Exception as e:
        print("Error creating subscription session:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# WEBHOOK: Update user subscription on payment
# ---------------------------
# after your existing stripe.api_key = ...
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

@app.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None),
):
    payload = await request.body()
    event = stripe.Webhook.construct_event(
        payload, stripe_signature, WEBHOOK_SECRET
    )

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        username = session["metadata"]["username"]
        mode     = session["mode"]  # "payment" or "subscription"

        # Decide sub type & expiry
        if mode == "subscription":
            sub_type = "monthly"
            expiry   = datetime.utcnow() + timedelta(days=30)
        else:
            sub_type = "one_time"
            expiry   = datetime.utcnow()

        # Write it into MongoDB
        await user_collection.update_one(
            {"username": username},
            {"$set": {
                "subscription": sub_type,
                "subscription_expiry": expiry
            }}
        )
    return {"status": "success"}

# ---------------------------
# UPDATE SUBSCRIPTION ENDPOINT (manually triggering update)
# ---------------------------
@app.post("/update-subscription")
async def update_subscription(subscription_type: str, current_user: dict = Depends(get_current_user)):
    expiry = determine_subscription_expiry(subscription_type)
    result = await user_collection.update_one(
        {"username": current_user["username"]},
        {"$set": {"subscription": subscription_type, "subscription_expiry": expiry}}
    )
    if result.modified_count == 1:
        print(f"DEBUG: Updated subscription for {current_user['username']}: {subscription_type}, expiry: {expiry}")
        updated_user = await user_collection.find_one({"username": current_user["username"]})
        print("DEBUG: Updated user data:", updated_user)
        return {"message": "Subscription updated successfully"}
    raise HTTPException(status_code=400, detail="Unable to update subscription")

# ---------------------------
# HELPER FUNCTION: Determine Subscription Expiry
# ---------------------------
def determine_subscription_expiry(subscription_type: str) -> Optional[datetime]:
    now = datetime.utcnow()
    if subscription_type == "monthly":
        return now + timedelta(days=30)
    elif subscription_type == "one_time":
        return now
    else:
        return None
