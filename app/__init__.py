# ---------------------------------------------------------------
# 1.  Imports & environment
# ---------------------------------------------------------------
import secrets, string
from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import bcrypt, jwt, os, uuid, json
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from bson import ObjectId
from pymongo import UpdateOne
import boto3, uuid, os, mimetypes
from werkzeug.utils import secure_filename
from bson.json_util import dumps, loads 
import datetime as dt
# from .ai_logic import validate_idea as ai_validate
from .ai_logic_v2 import validate_idea as ai_validate

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-south-1')
)
BUCKET = os.getenv('S3_BUCKET')

ses = boto3.client("ses", region_name="ap-south-1")   # or your region
SENDER = os.getenv('SENDERS_EMAIL')     

# def send_email(to: str, subject: str, body: str):
#     try:
#         ses.send_email(
#             Source=SENDER,
#             Destination={"ToAddresses": [to]},
#             Message={
#                 "Subject": {"Data": subject},
#                 "Body": {"Text": {"Data": body}},
#             },
#         )
#     except ClientError as e:
#         # Log the error but don’t block user creation
#         print("SES error:", e)

def send_email(to: str, subject: str, html_body: str, text_body: str = None):
    """
    Send email with HTML template support
    """
    try:
        message = {
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Html": {"Data": html_body, "Charset": "UTF-8"}
            }
        }
        
        # Add text fallback if provided
        if text_body:
            message["Body"]["Text"] = {"Data": text_body, "Charset": "UTF-8"}
            
        ses.send_email(
            Source=SENDER,
            Destination={"ToAddresses": [to]},
            Message=message
        )
        print(f"Welcome email sent successfully to {to}")
        
    except ClientError as e:
        print("SES error:", e)


def build_welcome_email(role: str, name: str, email: str, password: str) -> tuple[str, str]:
    """
    Return (subject, html_body) for the welcome e-mail
    tailored to the user’s role.
    """
    title_map = {
        "super_admin": "Super Administrator",
        "college_admin": "College Principal Admin",
        "ttc_coordinator": "TTC Coordinator",
        "innovator": "Innovator"
    }
    pretty_role = title_map.get(role, role.title())

    subject = f"Welcome to Pragathi Portal – Your {pretty_role} Account is Ready!"

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Welcome</title>
  <style>
    body {{
      margin:0; padding:0; font-family:Arial,Helvetica,sans-serif; background:#f7f7f7;
    }}
    .wrapper {{
      background:#f7f7f7; padding:40px 20px;
    }}
    .card {{
      max-width:600px; margin:0 auto; background:#ffffff; border-radius:12px;
      overflow:hidden; box-shadow:0 8px 25px rgba(0,0,0,.08);
    }}
    .header {{
      background:linear-gradient(135deg,#6366f1 0%, #3b82f6 100%);
      padding:30px; color:#ffffff; text-align:center;
    }}
    .header h1 {{
      margin:0; font-size:28px; font-weight:600;
    }}
    .body {{
      padding:30px; color:#333; line-height:1.6;
    }}
    .role-badge {{
      display:inline-block; background:#eef2ff; color:#3b82f6;
      padding:6px 14px; border-radius:20px; font-size:14px; font-weight:600;
      margin-bottom:20px;
    }}
    .credentials {{
      background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:20px;
      font-family:monospace; font-size:15px; color:#374151; margin:20px 0;
    }}
    .btn {{
      display:inline-block; background:#6366f1; color:#ffffff; padding:12px 24px;
      border-radius:8px; text-decoration:none; font-weight:600; margin-top:20px;
    }}
    .footer {{
      background:#f3f4f6; padding:20px; font-size:12px; color:#6b7280; text-align:center;
    }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="card">
      <div class="header">
        <h1>Welcome to Pragathi</h1>
      </div>
      <div class="body">
        <div class="role-badge">{pretty_role}</div>
        <p>Hi <strong>{name}</strong>,</p>
        <p>
          Your account has been created successfully. Below are your login
          credentials—please keep them safe and change your password after
          your first sign-in.
        </p>
        <div class="credentials">
          <strong>Email:</strong> {email}<br>
          <strong>Password:</strong> {password}
        </div>
      </div>
      <div class="footer">
        Need help? Reply to this email or visit our support portal.
      </div>
    </div>
  </div>
</body>
</html>
    """.strip()

    text = f"""
Welcome to Pragathi – {pretty_role}

Hi {name},

Your account has been created.

Email: {email}
Password: {password}

Please log in and change your password after first use.
Support: https://innosphere.com/support
    """.strip()

    return subject, html


# Example usage:
# send_welcome_email(
#     email="user@example.com",
#     user_name="John Doe", 
#     password="TempPass123!",
#     user_role="auditor"
# )




load_dotenv()

MONGO_URI     = os.getenv("MONGO_URI", "mongodb+srv://vikramvarma1201:XeRWARiS4Je1rXV0@cluster0.xsvkrgd.mongodb.net/pragati")
JWT_SECRET    = os.getenv("JWT_SECRET", "change-me-32-chars-minimum")
APP_ID        = os.getenv("PRAGATI_APP_ID", "default-pragati-app")

client = MongoClient(MONGO_URI)
db     = client.get_default_database()
users_coll      = db["users"]
ideas_coll      = db[f"{APP_ID}_ideas"]
credit_coll     = lambda cid: db[f"{APP_ID}_credit_requests_{cid}"]
program_coll    = db[f"{APP_ID}_ttc_programs"]

app = Flask(__name__)

# CORS(app, origins=["http://192.168.0.234:3000", "http://localhost:3000"],
#      supports_credentials=True) 

CORS(app,
     origins="*",          
     supports_credentials=True)

# ---------------------------------------------------------------
# 2.  Auth helpers
# ---------------------------------------------------------------
def hash_pwd(p):      return bcrypt.hashpw(p.encode(), bcrypt.gensalt())
def check_pwd(p, h):  return bcrypt.checkpw(p.encode(), h)

def create_token(uid, role):
    return jwt.encode({
        "uid": uid,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }, JWT_SECRET, algorithm="HS256")

def decode_token(tok):
    try:
        return jwt.decode(tok, JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return None
    

# clean doc helper

def clean_doc(doc):
    """
    Recursively convert:
      ObjectId -> str
      datetime -> ISO-8601 string
    """
    if isinstance(doc, dict):
        return {k: clean_doc(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [clean_doc(item) for item in doc]
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, datetime):
        return doc.isoformat()
    return doc

def parse_oid(oid_str, name="ID"):
    try:
        return ObjectId(oid_str)
    except (InvalidId, TypeError):
        raise ValueError(f"Invalid {name} format")
# ---------------------------------------------------------------
# 3.  Decorators
# ---------------------------------------------------------------
from functools import wraps

def requires_auth(f):
    @wraps(f)
    def _auth_wrapper(*args, **kwargs):
        tok = request.headers.get("Authorization", "").replace("Bearer ", "")
        payload = decode_token(tok)
        if not payload:
            return jsonify({"error": "Invalid or missing token"}), 401
        request.current_user = payload
        return f(*args, **kwargs)
    return _auth_wrapper

def requires_role(roles):
    def decorator(f):
        @requires_auth
        @wraps(f)
        def _role_wrapper(*args, **kwargs):
            role = request.current_user["role"]
            if role not in roles:
                return jsonify({"error": f"Access denied. Required: {roles}"}), 403
            return f(*args, **kwargs)
        return _role_wrapper
    return decorator

# ---------------------------------------------------------------
# 4.  Super-admin bootstrap
# ---------------------------------------------------------------
@app.route("/api/auth/super-admin/signup", methods=["POST"])
def super_admin_signup():
    """Create the first super-admin."""
    body = request.get_json(force=True)
    email, pwd = body.get("email"), body.get("password")
    if not email or not pwd:
        return jsonify({"error": "Email and password required"}), 400

    if users_coll.find_one({"role": "super_admin"}):
        return jsonify({"error": "Super-admin already exists"}), 409

    uid = str(uuid.uuid4())
    users_coll.insert_one({
        "_id": uid,
        "email": email,
        "password": hash_pwd(pwd),
        "role": "super_admin",
        "createdAt": datetime.now(timezone.utc),
        "createdBy": None
    })
    token = create_token(uid, "super_admin")
    return jsonify({"message": "Super-admin created", "token": token,"success":True}), 201

# ---------------------------------------------------------------
# 5.  Login (email + password)
# ---------------------------------------------------------------
@app.route("/api/auth/login", methods=["POST"])
def login():
    body = request.get_json(force=True)
    email, pwd = body.get("email"), body.get("password")
    user = users_coll.find_one({"email": email})
    if not user or not check_pwd(pwd, user["password"]):
        return jsonify({"error": "Invalid credentials"}), 401
    token = create_token(str(user["_id"]), user["role"])

    user_dict = {
        "uid": str(user["_id"]),
        "email": user["email"],
        "role": user["role"]
    }

    # only TTC coordinators get the collegeId in the response
    if user["role"] == "ttc_coordinator":
        user_dict["collegeId"] = user["collegeId"]

    return jsonify({"token": token, "user": user_dict, "success": True}), 200

# ---------------------------------------------------------------
# 6.  Super-admin creates other roles
# ---------------------------------------------------------------
# @app.route("/api/admin/create-user", methods=["POST"])
# @requires_role(["super_admin"])
# def create_user():
#     body = request.get_json(force=True)
#     email, pwd, role = body.get("email"), body.get("password"), body.get("role")
#     if role not in {"ttc_coordinator", "college_admin", "innovator"}:
#         return jsonify({"error": "Invalid role"}), 400
#     if users_coll.find_one({"email": email}):
#         return jsonify({"error": "Email already registered"}), 409

#     uid = str(uuid.uuid4())
#     users_coll.insert_one({
#         "_id": uid,
#         "email": email,
#         "password": hash_pwd(pwd),
#         "role": role,
#         "createdAt": datetime.now(timezone.utc),
#         "createdBy": request.current_user["uid"]
#     })
#     return jsonify({"message": f"{role} created", "uid": uid}), 201


# credit requst helper

def adjust_balance(uid, delta):
    """Add/subtract credits from the user's creditQuota."""
    res = users_coll.update_one(
        {"_id": uid},
        {"$inc": {"creditQuota": delta}}
    )
    return res.modified_count == 1

# innovator request for credit to ttc coordinator

@app.route("/api/credits/request-from-ttc", methods=["POST"])
@requires_role(["innovator"])
def innovator_credit_request():
    body = request.get_json(force=True)
    amount = int(body.get("amount", 0))
    reason = body.get("reason", "").strip()

    if amount <= 0 or not reason:
        return jsonify({"error": "amount>0 and reason required"}), 400

    # find the innovator’s TTC coordinator
    innovator = users_coll.find_one(
        {"_id": request.current_user["uid"]},
        {"ttcCoordinatorId": 1, "name": 1}
    )
    ttc_id = innovator.get("ttcCoordinatorId")
    if not ttc_id:
        return jsonify({"error": "TTC coordinator not linked"}), 400

    # create request document
    rid = str(uuid.uuid4())
    db[f"{APP_ID}_credit_requests_internal"].insert_one({
        "_id": rid,
        "from": request.current_user["uid"],
        "to": ttc_id,
        "amount": amount,
        "reason": reason,
        "status": "pending",
        "level": "innovator->ttc",
        "createdAt": datetime.now(timezone.utc)
    })
    return jsonify({"requestId": rid, "message": "Request sent to TTC","success":True}), 201

#get api for ttc to get list of requests

@app.route("/api/credits/ttc/incoming-requests", methods=["GET"])
@requires_role(["ttc_coordinator"])
def ttc_incoming_requests():
    """
    List all credit requests directed **to** the current TTC coordinator
    (i.e. from any of his/her innovators).
    """
    cursor = db[f"{APP_ID}_credit_requests_internal"].find(
        {
            "to": request.current_user["uid"],           # this TTC
            "level": "innovator->ttc"                    # only innovator→TTC flow
        },
        {"_id": 1, "from": 1, "amount": 1, "reason": 1,
         "status": 1, "createdAt": 1, "decidedAt": 1}
    ).sort("createdAt", -1)

    # enrich each record with the innovator’s name
    enriched = []
    for doc in cursor:
        innov = users_coll.find_one(
            {"_id": doc["from"]},
            {"name": 1, "email": 1}
        )
        doc["innovatorName"] = innov.get("name", "")
        doc["innovatorEmail"] = innov.get("email", "")
        enriched.append(doc)

    return jsonify({"success": True, "data": enriched}), 200

# -----------------------------------------------------------
#  TTC Coordinator : approve / reject credit request
# -----------------------------------------------------------
from pymongo import ReturnDocument

@app.route("/api/credits/ttc/incoming-requests/<rid>/decide", methods=["PUT"])
@requires_role(["ttc_coordinator"])
def ttc_decide_credit_request(rid):
    body = request.get_json(force=True)
    decision = body.get("decision")
    if decision not in {"approved", "rejected"}:
        return jsonify({"error": "decision must be 'approved' or 'rejected'"}), 400

    req_coll = db[f"{APP_ID}_credit_requests_internal"]
    req_doc = req_coll.find_one(
        {"_id": rid, "to": request.current_user["uid"], "status": "pending"}
    )
    if not req_doc:
        return jsonify({"error": "Request not found or already handled"}), 404

    ttc_id   = request.current_user["uid"]
    innov_id = req_doc["from"]
    amount   = req_doc["amount"]

    if decision == "rejected":
        req_coll.update_one(
            {"_id": rid},
            {"$set": {
                "status": "rejected",
                "decidedAt": datetime.now(timezone.utc),
                "decidedBy": ttc_id
            }}
        )
        return jsonify({"success": True, "message": "Request rejected"}), 200

    # ---- APPROVE FLOW ----
    # 1. Lock TTC doc and check / deduct credits
    ttc_doc = users_coll.find_one_and_update(
        {"_id": ttc_id, "creditQuota": {"$gte": amount}},
        {"$inc": {"creditQuota": -amount}},
        return_document=ReturnDocument.AFTER
    )

    if not ttc_doc:                # insufficient credits
        return jsonify({"error": "Not enough credits"}), 400

    # 2. Credit the innovator
    ok = users_coll.update_one(
        {"_id": innov_id},
        {"$inc": {"creditQuota": amount}}
    ).modified_count == 1

    if not ok:
        # Roll back TTC deduction
        users_coll.update_one(
            {"_id": ttc_id},
            {"$inc": {"creditQuota": amount}}
        )
        return jsonify({"error": "Failed to credit innovator"}), 500

    # 3. Mark request approved
    req_coll.update_one(
        {"_id": rid},
        {"$set": {
            "status": "approved",
            "decidedAt": datetime.now(timezone.utc),
            "decidedBy": ttc_id
        }}
    )

    return jsonify({"success": True, "message": "Request approved"}), 200
# ------------------------------------------------------------
# TTC Coordinator : request credits from College Admin
# ------------------------------------------------------------
@app.route("/api/credits/ttc/request-from-college", methods=["POST"])
@requires_role(["ttc_coordinator"])
def ttc_request_from_college():
    body = request.get_json(force=True)
    amount = int(body.get("amount", 0))
    reason = body.get("reason", "").strip()
    if amount <= 0 or not reason:
        return jsonify({"error": "amount>0 and reason required"}), 400

    rid = str(uuid.uuid4())
    db[f"{APP_ID}_credit_requests_internal"].insert_one({
        "_id": rid,
        "from": request.current_user["uid"],
        "to": None,                           # indicates "to college admin"
        "amount": amount,
        "reason": reason,
        "status": "pending",
        "level": "ttc->college",
        "createdAt": datetime.now(timezone.utc)
    })
    return jsonify({"requestId": rid, "message": "Request sent to college admin"}), 201

# ------------------------------------------------------------
# College Admin : list TTC credit requests
# ------------------------------------------------------------
@app.route("/api/credits/college/incoming-requests", methods=["GET"])
@requires_role(["college_admin","ttc_coordinator"])
def college_incoming_requests():
    cursor = db[f"{APP_ID}_credit_requests_internal"].find(
        {"level": "ttc->college"},
        {"_id": 1, "from": 1, "amount": 1, "reason": 1,
         "status": 1, "createdAt": 1}
    ).sort("createdAt", -1)

    enriched = []
    for doc in cursor:
        ttc = users_coll.find_one({"_id": doc["from"]}, {"name": 1, "email": 1})
        doc["ttcName"] = ttc.get("name", "")
        doc["ttcEmail"] = ttc.get("email", "")
        enriched.append(doc)
    return jsonify({"data": enriched}), 200

# ------------------------------------------------------------
# College Admin : approve / reject TTC request
# ------------------------------------------------------------
@app.route("/api/credits/college/incoming-requests/<rid>/decide", methods=["PUT"])
@requires_role(["college_admin"])
def college_decide_ttc_request(rid):
    body = request.get_json(force=True)
    decision = body.get("decision")
    if decision not in {"approved", "rejected"}:
        return jsonify({"error": "Invalid decision"}), 400

    req_coll = db[f"{APP_ID}_credit_requests_internal"]
    req = req_coll.find_one(
        {"_id": rid, "level": "ttc->college", "status": "pending"}
    )
    if not req:
        return jsonify({"error": "Request not found"}), 404

    amount   = req["amount"]
    ttc_id   = req["from"]
    admin_id = request.current_user["uid"]

    # --- decision logic ---
    if decision == "rejected":
        req_coll.update_one(
            {"_id": rid},
            {"$set": {"status": "rejected", "decidedAt": datetime.now(timezone.utc)}}
        )
        return jsonify({"message": "Request rejected"}), 200

    # --- APPROVE FLOW ---
    # 1. Verify admin has enough credits
    admin_doc = users_coll.find_one({"_id": admin_id}, {"creditQuota": 1})
    if not admin_doc or admin_doc.get("creditQuota", 0) < amount:
        return jsonify({"error": "Insufficient college credits"}), 400

    # 2. Atomic deduction from admin + addition to TTC
    res = users_coll.bulk_write([
        UpdateOne(
            {"_id": admin_id, "creditQuota": {"$gte": amount}},
            {"$inc": {"creditQuota": -amount}}
        ),
        UpdateOne(
            {"_id": ttc_id},
            {"$inc": {"creditQuota": amount}}
        ),
    ])

    if res.modified_count != 2:   # either admin or TTC update failed
        return jsonify({"error": "Failed to transfer credits"}), 500

    # 3. Mark request approved
    req_coll.update_one(
        {"_id": rid},
        {"$set": {"status": "approved", "decidedAt": datetime.now(timezone.utc)}}
    )

    return jsonify({"message": "Request approved"}), 200


@app.route("/api/credits/my-pending-request/<user_id>", methods=["GET"])
@requires_auth
def get_my_pending_request(user_id):
    coll = db[f"{APP_ID}_credit_requests_internal"]
    doc = coll.find_one(
        {"from": user_id, "status": "pending"},
        sort=[("createdAt", -1)]  # newest first
    )

    if not doc:
        return jsonify({"success": True, "data": None}), 200

    # enrich
    from_user = users_coll.find_one({"_id": doc["from"]}, {"name": 1, "email": 1})
    to_user = None
    if doc["to"]:
        to_user = users_coll.find_one({"_id": doc["to"]}, {"name": 1, "email": 1})

    doc["fromName"] = from_user.get("name", "") if from_user else ""
    doc["fromEmail"] = from_user.get("email", "") if from_user else ""
    doc["toName"] = to_user.get("name", "") if to_user else ""
    doc["toEmail"] = to_user.get("email", "") if to_user else ""

    return jsonify({"success": True, "data": clean_doc(doc)}), 200

@app.route("/api/credits/<request_id>", methods=["DELETE"])
@requires_auth
def delete_credit_request(request_id):
    coll = db[f"{APP_ID}_credit_requests_internal"]
    res = coll.delete_one({"_id": request_id, "from": request.current_user["uid"]})
    if res.deleted_count == 0:
        return jsonify({"error": "Request not found or not yours"}), 404
    return jsonify({"success": True, "message": "Request deleted"}), 200

# -----------------------------------------------------------
#  A.  Super-admin : create College Principal Admin
# -----------------------------------------------------------
@app.route("/api/admin/create-principal", methods=["POST"])
@requires_role(["super_admin"])
def create_principal_admin():
    body = request.get_json(force=True)
    college_name = body.get("collegeName")
    email        = body.get("email")
    ttc_limit    = body.get("ttcCoordinatorLimit", 0)
    credit_quota = body.get("creditQuota", 0)

    if not college_name or not email:
        return jsonify({"error": "collegeName & email required"}), 400
    if users_coll.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409

    # generate random 10-char password
    pwd_plain = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
    # pwd_plain="123456"
    pwd_hash  = hash_pwd(pwd_plain)

    uid = str(uuid.uuid4())
    users_coll.insert_one({
        "_id"              : uid,
        "email"            : email,
        "password"         : pwd_hash,
        "role"             : "college_admin",
        "collegeName"      : college_name,
        "ttcCoordinatorLimit": int(ttc_limit),
        "creditQuota"      : int(credit_quota),
        "createdAt"        : datetime.now(timezone.utc),
        "createdBy"        : request.current_user["uid"],
        "isActive":True,
        "isDeleted":False,
    })

    # send credential mail
    subject, html = build_welcome_email("college_admin", college_name, email, pwd_plain)
    send_email(email, subject, html)
    # send_ses_email(
    #     to=email,
    #     subject=f"Welcome {college_name} – Principal Admin Credentials",
    #     body=f"Login email: {email}\nPassword: {pwd_plain}\nPlease change your password after first login."
    # )
    return jsonify({"message": "Principal admin created", "uid": uid}), 201


# -----------------------------------------------------------
#  B.  College Principal Admin : create TTC Coordinator
# -----------------------------------------------------------
@app.route("/api/principal/create-coordinator", methods=["POST"])
@requires_role(["college_admin"])
def create_ttc_coordinator():
    body = request.get_json(force=True)
    name  = body.get("name")
    email = body.get("email")
    expertise = body.get("expertise")  # comma-separated string

    if not name or not email:
        return jsonify({"error": "name & email required"}), 400
    if users_coll.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409

    # check limit
    principal = users_coll.find_one({"_id": request.current_user["uid"]})
    current   = users_coll.count_documents({"createdBy": request.current_user["uid"], "role": "ttc_coordinator"})
    if current >= principal.get("ttcCoordinatorLimit", 0):
        return jsonify({"error": "TTC coordinator limit reached"}), 409

    pwd_plain = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
    # pwd_plain="123456"
    pwd_hash  = hash_pwd(pwd_plain)

    uid = str(uuid.uuid4())
    users_coll.insert_one({
        "_id"        : uid,
        "email"      : email,
        "password"   : pwd_hash,
        "role"       : "ttc_coordinator",
        "name"       : name,
        "expertise"  : [x.strip() for x in expertise.split(",")],
        "collegeId"  : request.current_user["uid"],   # points to principal
        "createdAt"  : datetime.now(timezone.utc),
        "createdBy"  : request.current_user["uid"],
        "isActive":True,
        "isDeleted":False,
        "creditQuota":0
    })

    subject, html = build_welcome_email("ttc_coordinator", name, email, pwd_plain)
    send_email(email, subject, html)

    # send_email(
    #     to=email,
    #     subject="Welcome – TTC Coordinator Credentials",
    #     body=f"Login email: {email}\nPassword: {pwd_plain}\nPlease change your password after first login."
    # )
    return jsonify({"message": "TTC coordinator created", "uid": uid}), 201


# -----------------------------------------------------------
#  C1. Principal : create single Mentor
# -----------------------------------------------------------
@app.route("/api/principal/create-mentor", methods=["POST"])
@requires_role(["college_admin"])
def create_mentor():
    body = request.get_json(force=True)
    name  = body.get("name")
    email = body.get("email")
    expertise_raw = body.get("expertise", "")          # comma-separated string

    if not name or not email:
        return jsonify({"error": "name & email required"}), 400
    if users_coll.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409

    # generate 10-char temp password
    # pwd_plain = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
    pwd_plain = "123456"
    pwd_hash  = hash_pwd(pwd_plain)

    uid = str(uuid.uuid4())
    users_coll.insert_one({
        "_id"        : uid,
        "email"      : email,
        "password"   : pwd_hash,
        "role"       : "mentor",
        "name"       : name,
        "expertise"  : [x.strip() for x in expertise_raw.split(",") if x.strip()],
        "collegeId"  : request.current_user["uid"],   # principal’s college
        "createdAt"  : datetime.now(timezone.utc),
        "createdBy"  : request.current_user["uid"],
        "isActive"   : True,
        "isDeleted"  : False,
        "creditQuota": 0
    })

    # subject, html = build_welcome_email("mentor", name, email, pwd_plain)
    # send_email(email, subject, html)
    return jsonify({"message": "Mentor created", "uid": uid}), 201


# -----------------------------------------------------------
#  C2. Principal : bulk upload Mentors (CSV / Excel)
# -----------------------------------------------------------
ALLOWED_MENTOR_MIME = {"text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}

@app.route("/api/mentors/bulk", methods=["POST"])
@requires_role(["college_admin"])
def bulk_upload_mentors():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file.content_type not in ALLOWED_MENTOR_MIME:
        return jsonify({"error": "Only .csv or .xlsx allowed"}), 400

    # read into pandas
    try:
        import pandas as pd
        if file.content_type == "text/csv":
            df = pd.read_csv(file.stream)
        else:
            df = pd.read_excel(file.stream)
    except Exception as e:
        return jsonify({"error": f"Cannot parse file: {str(e)}"}), 400

    required = {"name", "email", "expertise"}
    if not required.issubset(df.columns):
        return jsonify({"error": f"Missing columns: {required - set(df.columns)}"}), 400

    df = df.dropna(subset=["name", "email"])
    inserted, failed = [], []

    for _, row in df.iterrows():
        try:
            name, email, expertise_raw = row["name"].strip(), row["email"].strip(), str(row["expertise"])
            if users_coll.find_one({"email": email}):
                failed.append({"email": email, "reason": "Email already exists"})
                continue

            # pwd_plain = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
            pwd_plain = "123456"
            uid = str(uuid.uuid4())
            users_coll.insert_one({
                "_id"        : uid,
                "email"      : email,
                "password"   : hash_pwd(pwd_plain),
                "role"       : "mentor",
                "name"       : name,
                "expertise"  : [x.strip() for x in expertise_raw.split(",") if x.strip()],
                "collegeId"  : request.current_user["uid"],
                "createdAt"  : datetime.now(timezone.utc),
                "createdBy"  : request.current_user["uid"],
                "isActive"   : True,
                "isDeleted"  : False,
                "creditQuota": 0
            })
            inserted.append(email)
            # fire-and-forget welcome e-mail
            # subject, html = build_welcome_email("mentor", name, email, pwd_plain)
            # send_email(email, subject, html)
        except Exception as e:
            failed.append({"email": email, "reason": str(e)})

    return jsonify({
        "message": f"{len(inserted)} mentors created, {len(failed)} failed",
        "inserted": inserted,
        "failed": failed
    }), 200

# -----------------------------------------------------------
#  C3. List all Mentors belonging to the caller’s college
#      (innovator & ttc_coordinator may also read)
# -----------------------------------------------------------
@app.route("/api/mentors", methods=["GET"])
@requires_role(["college_admin", "ttc_coordinator", "innovator"])
def list_college_mentors():
    caller = request.current_user
    caller_id   = caller["uid"]
    caller_role = caller["role"]

    # 1. Resolve the college id
    if caller_role == "college_admin":
        college_id = caller_id
    else:
        # ttc or innovator – both have collegeId in their doc
        user_doc = users_coll.find_one({"_id": caller_id}, {"collegeId": 1})
        if not user_doc or "collegeId" not in user_doc:
            return jsonify({"error": "College not found for user"}), 400
        college_id = user_doc["collegeId"]

    # 2. Build query
    query = {
        "collegeId": college_id,
        "role": "mentor",
        "isDeleted": {"$ne": True}
    }
    projection = {"password": 0}

    mentors = list(users_coll.find(query, projection).sort("createdAt", -1))
    return jsonify({"success": True, "data": clean_doc(mentors)}), 200

# -----------------------------------------------------------
#  C.  TTC Coordinator : create Innovator
# -----------------------------------------------------------
@app.route("/api/coordinator/create-innovator", methods=["POST"])
@requires_role(["ttc_coordinator"])
def create_innovator():
    body = request.get_json(force=True)
    name  = body.get("name")
    email = body.get("email")

    if not name or not email:
        return jsonify({"error": "name & email required"}), 400
    if users_coll.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409

    # fetch coordinator record to get the linked collegeId
    coordinator = users_coll.find_one(
        {"_id": request.current_user["uid"]},
        {"collegeId": 1}
    )
    if not coordinator or "collegeId" not in coordinator:
        return jsonify({"error": "Coordinator college not found"}), 400

    # temp password
    pwd_plain = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))

    # pwd_plain = "123456"
    pwd_hash  = hash_pwd(pwd_plain)

    uid = str(uuid.uuid4())
    users_coll.insert_one({
        "_id"       : uid,
        "email"     : email,
        "password"  : pwd_hash,
        "role"      : "innovator",
        "name"      : name,
        "collegeId" : coordinator["collegeId"],
        "ttcCoordinatorId" : request.current_user["uid"],
        "createdAt" : datetime.now(timezone.utc),
        "createdBy" : request.current_user["uid"],
        "isActive":True,
        "isDeleted":False,
        "creditQuota":0
    })

    subject, html = build_welcome_email("Innovator", name, email, pwd_plain)
    send_email(email, subject, html)
    
    # send_email(
    #     to=email,
    #     subject="Welcome – Innovator",
    #     body=f"Login email: {email}\nPassword: {pwd_plain}\nPlease change your password after first login."
    # )

    # send_email(...)
    return jsonify({"message": "Innovator created", "uid": uid}), 201

# ------------------------------------------------------------------
# GET /api/innovators
# Returns every innovator (role = "innovator") that is NOT soft-deleted.
# Accessible by: ttc_coordinator, college_admin, super_admin
# ------------------------------------------------------------------
@app.route("/api/innovators", methods=["GET"])
@requires_role(["ttc_coordinator", "college_admin", "super_admin"])
def get_all_innovators():
    q = {"role": "innovator", "isDeleted": {"$ne": True}}
    cursor = users_coll.find(q, {"password": 0}).sort("createdAt", -1)
    return jsonify({"success": True, "data": list(cursor)}), 200

# -----------------------------------------------------------
#  D.  Single login endpoint for every role
# -----------------------------------------------------------
#  Soft-delete a user
@app.route("/api/users/<uid>/soft-delete", methods=["PUT"])
@requires_role(["super_admin", "college_admin", "ttc_coordinator"])
def soft_delete_user(uid):
    res = users_coll.update_one(
        {"_id": uid},
        {"$set": {"isDeleted": True}}
    )
    if res.matched_count == 0:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"message": "User soft-deleted"}), 200


# Toggle isActive (true ↔ false)
@app.route("/api/users/<uid>/toggle-active", methods=["PUT"])
@requires_role(["super_admin", "college_admin", "ttc_coordinator"])
def toggle_active(uid):
    user = users_coll.find_one({"_id": uid}, {"_id": 1, "isActive": 1})
    if not user:
        return jsonify({"error": "User not found"}), 404
    new_val = not user.get("isActive", True)
    users_coll.update_one({"_id": uid}, {"$set": {"isActive": new_val}})
    return jsonify({"message": f"isActive set to {new_val}", "isActive": new_val}), 200

# Update User
@app.route("/api/users/<uid>", methods=["PUT", "PATCH"])
@requires_auth
def update_user(uid):
    caller = request.current_user
    caller_id   = caller["uid"]
    caller_role = caller["role"]
    caller_college = caller.get("collegeId")  # college_admin / ttc_coordinator

    # 1. Load target
    target = users_coll.find_one({"_id": uid, "isDeleted": {"$ne": True}})
    if not target:
        return jsonify({"error": "User not found"}), 404

    target_role  = target["role"]
    target_college = target.get("collegeId")

    # 2. Authorisation matrix
    ok = False
    if caller_role == "super_admin":
        ok = True
    elif caller_role == "college_admin":
        # can edit self, any TTC or innovator under their college
        ok = (uid == caller_id) or target_college == caller_college
    elif caller_role == "ttc_coordinator":
        # can edit self, any innovator they created
        ok = (uid == caller_id) or (target_role == "innovator" and target.get("createdBy") == caller_id)
    elif caller_role == "innovator":
        ok = uid == caller_id
    else:
        return jsonify({"error": "Access denied"}), 403

    if not ok:
        return jsonify({"error": "You cannot edit this user"}), 403

    # 3. Handle payload & file upload (unchanged from previous snippet)
    payload = {}
    if request.is_json:
        payload = request.get_json(force=True)
    elif request.content_type and request.content_type.startswith("multipart/"):
        payload = json.loads(request.form.get("json", "{}"))

    image_url = None
    if "image" in request.files:
        file = request.files["image"]
        if file and file.filename:
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext not in {"png", "jpg", "jpeg", "gif"}:
                return jsonify({"error": "Unsupported image type"}), 400
            key = f"profile-images/{uid}/{uuid.uuid4()}.{ext}"
            s3.upload_fileobj(
                file,
                BUCKET,
                key,
                ExtraArgs={"ContentType": mimetypes.types_map.get(f".{ext}", "application/octet-stream")},
            )
            image_url = f"https://{BUCKET}.s3.amazonaws.com/{key}"
            payload["profileImageUrl"] = image_url

    protected = {"_id", "role", "createdBy", "createdAt", "isDeleted"}
    updates = {k: v for k, v in payload.items() if k not in protected}

    if "email" in updates:
        if users_coll.find_one({"email": updates["email"], "_id": {"$ne": uid}}):
            return jsonify({"error": "Email already in use"}), 409

    updates["updatedAt"] = datetime.now(timezone.utc)
    users_coll.update_one({"_id": uid}, {"$set": updates})

    return jsonify({
        "message": "User updated",
        "profileImageUrl": image_url or target.get("profileImageUrl"),
    }), 200

# change password
@app.route("/api/users/<uid>/password", methods=["PUT"])
@requires_role(["ttc_coordinator", "college_admin", "super_admin", "innovator"])
def update_password(uid):
    body = request.get_json(force=True)
    current_password = body.get("currentPassword")
    new_password     = body.get("newPassword")

    # Validate input
    if not current_password or not new_password:
        return jsonify({"error": "Both current and new passwords are required"}), 400

    # Ensure user is changing their own password
    if uid != request.current_user["uid"]:
        return jsonify({"error": "Unauthorized to change another user's password"}), 403

    # Fetch user
    user = users_coll.find_one({"_id": uid})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Check if current password is correct
    if not check_pwd(current_password, user.get("password", "")):
        return jsonify({"error": "Current password is incorrect"}), 401

    # Hash new password
    new_password_hash = hash_pwd(new_password)

    # Update in DB
    users_coll.update_one(
        {"_id": uid},
        {"$set": {"password": new_password_hash}}
    )

    return jsonify({"message": "Password updated successfully"}), 200


#  List users with optional filters
@app.route("/api/users", methods=["GET"])
@requires_role(["super_admin", "college_admin", "ttc_coordinator"])
def list_users():
    role_filter      = request.args.get("role")          # optional
    college_id       = request.args.get("college_id")    # optional
    ttc_id           = request.args.get("ttc_id")        # optional

    query = {"isDeleted": {"$ne": True}}                 # never show soft-deleted

    if role_filter:              # e.g. ?role=college_admin
        query["role"] = role_filter
    if college_id and not ttc_id:
        query.update({"role": "ttc_coordinator", "collegeId": college_id})
    if college_id and ttc_id:
        query.update({"role": "innovator", "collegeId": college_id, "createdBy": ttc_id})

    cursor = users_coll.find(query, {"password": 0})
    return jsonify({"docs":list(cursor),"success":True}), 200


#  Single user by _id
@app.route("/api/users/<uid>", methods=["GET"])
@requires_role(["super_admin", "college_admin", "ttc_coordinator","innovator"])
def get_user(uid):
    user = users_coll.find_one({"_id": uid}, {"password": 0})
    if not user or user.get("isDeleted"):
        return jsonify({"error": "User not found"}), 404
    return jsonify(user), 200

# ---------------------------------------------------------------
# 7.  Protected routes (ideas, programs, credit requests)
# ---------------------------------------------------------------
# ALLOWED_EXT = {'ppt', 'pptx'}
# @app.route("/api/ideas/validate", methods=["POST"])
# @requires_role(["innovator", "college_admin", "super_admin"])
# def validate_idea():
#     user_id = request.current_user['uid']
#     user_doc = users_coll.find_one(
#         {'_id': user_id},
#         {'creditQuota': 1, 'noOfIdeas': 1, 'ttcCoordinatorId': 1, 'collegeId': 1}
#     )
#     if not user_doc:
#         return jsonify({'error': 'User record not found'}), 404

#     if user_doc.get('creditQuota', 0) <= 0:
#         return jsonify({'error': 'No credits remaining. Please purchase more.'}), 402 
#     # ---------- 1.  File ----------
#     if 'pptFile' not in request.files:
#         return jsonify({"error": "pptFile is required"}), 400
#     file = request.files['pptFile']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400
#     ext = file.filename.rsplit('.', 1)[-1].lower()
#     if ext not in ALLOWED_EXT:
#         return jsonify({"error": "Only .ppt or .pptx allowed"}), 400

#     # ---------- 2.  JSON ----------
#     name        = request.form.get("ideaName")
#     concept     = request.form.get("ideaConcept")
#     domain      = request.form.get("domain")
#     preset      = request.form.get("preset")
#     weights     = {k: int(request.form.get(k, 0)) for k in [
#         "Core Idea & Innovation",
#         "Market & Commercial Opportunity",
#         "Execution & Operations",
#         "Business Model & Strategy",
#         "Team & Organizational Health",
#         "External Environment & Compliance",
#         "Risk & Future Outlook"
#     ]}

#     if not name or not concept:
#         return jsonify({"error": "ideaName & ideaConcept required"}), 400

#     # ---------- 3.  AI ----------
#     try:
#         res = ai_validate(name, concept, weights)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     if res.get("error"):
#         return jsonify({"error": res["error"]}), 500

#     # ---------- 4.  S3 ----------
#     # user_id  = request.current_user['uid']
#     # user_doc = users_coll.find_one({'_id': user_id}, {'ttcCoordinatorId': 1, 'collegeId': 1})
#     # if not user_doc:
#     #     return jsonify({'error': 'User record not found'}), 404

#     file_key = f"ideas/{user_id}/{uuid.uuid4()}.{ext}"
#     s3.upload_fileobj(
#         file,
#         BUCKET,
#         file_key,
#         ExtraArgs={'ContentType': mimetypes.types_map.get(f'.{ext}', 'application/octet-stream')}
#     )
#     s3_url = f"https://{BUCKET}.s3.amazonaws.com/{file_key}"

#     score = float(res["overall_score"])
#     if score < 1.5:
#         status = "rejected"
#     elif score < 3.5:
#         status = "improvise"   # or any wording you prefer
#     else:
#         status = "accepted"


#     # ---------- 5.  Mongo ----------
#     doc = {
#         "_id"             : str(uuid.uuid4()),
#         "userId"          : user_id,
#         "ideaName"        : name,
#         "ideaConcept"     : concept,
#         "overallScore"    : res["overall_score"],
#         "status"           : status,  
#         "validationOutcome": res["validation_outcome"],
#         "evaluatedData"   : res["evaluated_data"],
#         "htmlReport"      : res["html_report"],
#         "pptUrl"          : s3_url,
#         "weights"         : weights,
#         "preset"          : preset,
#         "domain"          : domain,
#         "ttcCoordinatorId": user_doc.get('ttcCoordinatorId'),
#         "collegeId"       : user_doc.get('collegeId'),
#         "createdAt"       : datetime.now(timezone.utc),
#         "updatedAt"       : datetime.now(timezone.utc),
#         "isDeleted"       :False,
#     }
#     ideas_coll.insert_one(doc)

#     users_coll.update_one(
#         {'_id': user_id},
#         {
#             '$inc': {
#                 'creditQuota': -1,
#             'noOfIdeas':    1
#             }
#         }
#     )

#     return jsonify({
#         "success": True,
#         "message": "Idea saved",
#         "ideaId": doc["_id"],
#         "score": doc["overallScore"],
#         "pptUrl": s3_url
#     }), 201

# ------------------------------------------------------------------
#  NEW IDEA DRAFT + TEAM + MENTOR + UPLOAD + SUBMIT  FLOW
# ------------------------------------------------------------------
drafts_coll = db[f"{APP_ID}_ideas_draft"]     # <- add this line near the top

@app.route("/api/ideas/draft", methods=["POST"])
@requires_role(["innovator"])
def save_draft():
    uid   = request.current_user["uid"]
    data  = request.get_json(force=True)
    draft_id = data.pop("draftId", None)

    # 1.  ---  handle core-team resolution  ---
    core_emails = data.pop("coreTeamEmails", None) or []
    core_oids   = []                       # we will store ObjectIds here
    if core_emails:                        # only if client sent the key
        # normalise to lower-case
        core_emails = [e.lower().strip() for e in core_emails]
        # bulk-fetch existing users with compatible roles
        found = users_coll.find(
            {"email": {"$in": core_emails}, "role": {"$in": ["innovator", "team_member"]}},
            {"_id": 1, "email": 1}
        )
        email_to_id = {doc["email"]: doc["_id"] for doc in found}
        core_oids   = [email_to_id[e] for e in core_emails if e in email_to_id]

    # 2.  ---  build payload  ---
    payload = {
        "updatedAt": datetime.now(timezone.utc),
        **data,
        "coreTeamIds": core_oids          # replace entire list (idempotent)
    }

    # 3.  ---  insert vs update (unchanged logic)  ---
    if draft_id:
        try:
            oid = parse_oid(draft_id, "draftId")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        res = drafts_coll.update_one(
            {"_id": oid, "ownerId": uid},
            {"$set": payload}
        )
        if res.matched_count == 0:
            return jsonify({"error": "Draft not found"}), 404
        out_id = draft_id
    else:
        payload["ownerId"] = uid
        payload["_id"]     = ObjectId()
        drafts_coll.insert_one(payload)
        out_id = str(payload["_id"])

    # 4.  ---  response  ---
    return jsonify({
        "draftId": out_id,
        "coreTeamIds": [str(oid) for oid in core_oids],  
        "notFoundEmails": [e for e in core_emails if e not in email_to_id] 
    }), 200

# ---------- 2.  INVITE TEAM MEMBER ----------
# ----------  INVITE CORE-TEAM MEMBER  ----------
@app.route("/api/ideas/draft/team", methods=["POST"])
@requires_role(["innovator"])
def invite_team_member():
    uid = request.current_user["uid"]
    data = request.get_json(force=True)
    try:
        draft_id, email = parse_oid(data["draftId"], "draftId"), data["email"].lower()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    draft = drafts_coll.find_one({"_id": draft_id, "ownerId": uid})
    if not draft:
        return jsonify({"error": "Draft not found"}), 404

    # 1. ----  EXISTING USER  ----  NO-OP  ----
    existing = users_coll.find_one({"email": email})
    if existing:
        # We do NOT touch the draft or the existing user
        return jsonify({
            "userId": str(existing["_id"]),
            "status": existing.get("status"),
            "wasCreated": False,
            "message": "User already exists – no invitation sent"
        }), 200

    # 2. ----  CREATE INVITED team_member  ----
    user_id = ObjectId()
    pwd_plain = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
    users_coll.insert_one({
        "_id"        : user_id,
        "email"      : email,
        "password"   : hash_pwd(pwd_plain),   # temporary, force change later
        "role"       : "team_member",
        "status"     : "invited",
        "invitedBy"  : uid,
        "collegeId"  : request.current_user.get("collegeId"),
        "createdAt"  : datetime.now(timezone.utc),
        "isActive"   : True,
        "isDeleted"  : False,
        "creditQuota": 0
    })

    # 3. ----  SEND INVITATION E-MAIL  ----
    token = create_token(str(user_id), "team_member")   
    accept_link = f"{request.host_url}api/invites/team/accept?token={token}&draftId={str(draft_id)}"
    reject_link = f"{request.host_url}api/invites/team/reject?token={token}&draftId={str(draft_id)}"

    html_body = f"""
    <p>You have been invited to join an innovation team.</p>
    <p><a href="{accept_link}">Accept</a> | <a href="{reject_link}">Reject</a></p>
    <p>Your temporary password is: <strong>{pwd_plain}</strong><br>
    Please change it after first login.</p>
    """
    send_email(email, "Invitation to join innovation team", html_body)

    return jsonify({
        "userId"     : str(user_id),
        "status"     : "invited",
        "wasCreated" : True,
        "message"    : "Invitation sent"
    }), 201

#   accept or reject invitaion

@app.route("/api/invites/team/<action>", methods=["GET"])
def team_decide(action):
    if action not in ("accept", "reject"):
        return jsonify({"error": "Bad action"}), 400
    tok = request.args.get("token")
    draft_id_str = request.args.get("draftId")
    payload = decode_token(tok)
    if not payload or payload.get("role") != "team_member":
        return jsonify({"error": "Invalid or expired token"}), 400

    uid, did = parse_oid(payload["uid"], "userId"), parse_oid(draft_id_str, "draftId")

    if action == "accept":
        # promote status + add to team
        users_coll.update_one({"_id": uid}, {"$set": {"status": "active"}})
        drafts_coll.update_one({"_id": did}, {"$addToSet": {"coreTeamIds": uid}})
        return jsonify({"status": "accepted"}), 200

    else:  # reject
        # hard-delete the invited account
        users_coll.delete_one({"_id": uid, "role": "team_member", "status": "invited"})
        return jsonify({"status": "rejected"}), 200

# ---------- 3.  TEAM DECIDE ----------
# @app.route("/api/invites/team/<action>", methods=["GET"])
# def team_decide(action):
#     if action not in ("accept", "reject"):
#         return jsonify({"error": "Bad action"}), 400
#     tok = request.args.get("token")
#     payload = decode_token(tok)
#     if not payload:
#         return jsonify({"error": "Invalid or expired token"}), 400
#     user_id, draft_id = payload["uid"], payload.get("draftId")

#     try:
#         uid, did = parse_oid(user_id, "userId"), parse_oid(draft_id, "draftId")
#     except ValueError as e:
#         return jsonify({"error": str(e)}), 400

#     if action == "accept":
#         users_coll.update_one({"_id": uid}, {"$set": {"status": "active"}})
#     else:
#         users_coll.delete_one({"_id": uid, "role": "team_member", "status": "invited"})
#         drafts_coll.update_one({"_id": did}, {"$pull": {"coreTeamIds": uid}})

#     return jsonify({"status": action}), 200

# ---------- 4.  ASSIGN MENTOR ----------
@app.route("/api/ideas/draft/mentor", methods=["POST"])
@requires_role(["innovator"])
def assign_mentor():
    uid = request.current_user["uid"]
    data = request.get_json(force=True)
    try:
        draft_id, mentor_id = parse_oid(data["draftId"], "draftId"), parse_oid(data["mentorId"], "mentorId")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    mentor = users_coll.find_one({"_id": mentor_id, "role": "mentor"})
    if not mentor:
        return jsonify({"error": "Invalid mentor"}), 400

    res = drafts_coll.update_one(
        {"_id": draft_id, "ownerId": uid},
        {"$set": {"mentorId": mentor_id, "mentorStatus": "pending"}}
    )
    if res.matched_count == 0:
        return jsonify({"error": "Draft not found"}), 404

    token = create_token(str(mentor_id), "mentor", draft_id=str(draft_id))
    link_accept = f"{request.host_url}api/invites/mentor/accept?token={token}&draftId={str(draft_id)}"
    link_reject = f"{request.host_url}api/invites/mentor/reject?token={token}&draftId={str(draft_id)}"

    return jsonify({"mentorStatus": "pending", "inviteLink": link_accept}), 200

# ---------- 5.  MENTOR DECIDE ----------
@app.route("/api/invites/mentor/<action>", methods=["GET"])
def mentor_decide(action):
    if action not in ("accept", "reject"):
        return jsonify({"error": "Bad action"}), 400
    tok = request.args.get("token")
    payload = decode_token(tok)
    if not payload:
        return jsonify({"error": "Invalid or expired token"}), 400
    mentor_id, draft_id = payload["uid"], request.args.get("draftId")

    try:
        mid, did = parse_oid(mentor_id, "mentorId"), parse_oid(draft_id, "draftId")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    new_status = "accepted" if action == "accept" else "rejected"
    drafts_coll.update_one(
        {"_id": did, "mentorId": mid},
        {"$set": {"mentorStatus": new_status}}
    )
    return jsonify({"mentorStatus": new_status}), 200

# ---------- 6.  UPLOAD PPT ----------
@app.route("/api/ideas/draft/upload", methods=["POST"])
@requires_role(["innovator"])
def upload_draft_ppt():
    uid = request.current_user["uid"]
    draft_id_str = request.form.get("draftId")
    try:
        draft_id = parse_oid(draft_id_str, "draftId")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if "pptFile" not in request.files:
        return jsonify({"error": "pptFile required"}), 400

    file = request.files["pptFile"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = secure_filename(file.filename).rsplit(".", 1)[-1].lower()
    if ext not in {"ppt", "pptx"}:
        return jsonify({"error": "Only .ppt or .pptx allowed"}), 400

    file.seek(0, os.SEEK_END)
    if file.tell() > 10 * 1024 * 1024:
        return jsonify({"error": "File too large (>10 MB)"}), 413
    file.seek(0)

    key = f"drafts/{uid}/{uuid.uuid4()}.{ext}"
    # upload_to_s3(file, key)  # your existing helper
    s3.upload_fileobj(
        file,
        BUCKET,
        key,
        ExtraArgs={'ContentType': mimetypes.types_map.get(f'.{ext}', 'application/octet-stream')}
    )
    s3_url = f"https://{BUCKET}.s3.amazonaws.com/{key}"

    drafts_coll.update_one(
        {"_id": draft_id, "ownerId": uid},
        {"$set": {"pptFileKey": key}}
    )
    return jsonify({"pptFileKey": key}), 200

# ---------- 7.  FINAL SUBMIT ----------
@app.route("/api/ideas/draft/submit", methods=["POST"])
@requires_role(["innovator"])
def submit_draft():
    uid = request.current_user["uid"]
    data = request.get_json(force=True)
    try:
        draft_id = parse_oid(data["draftId"], "draftId")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    draft = drafts_coll.find_one({"_id": draft_id, "ownerId": uid})
    if not draft:
        return jsonify({"error": "Draft not found"}), 404

    if draft.get("mentorStatus") != "accepted":
        return jsonify({"error": "Mentor must accept invitation first"}), 400
    if not draft.get("pptFileKey"):
        return jsonify({"error": "Please upload your pitch deck first"}), 400

    user_doc = users_coll.find_one_and_update(
        {"_id": uid, "creditQuota": {"$gte": 1}},
        {"$inc": {"creditQuota": -1, "noOfIdeas": 1}},
        return_document=ReturnDocument.AFTER,
        projection={"ttcCoordinatorId": 1, "collegeId": 1}
    )
    if not user_doc:
        return jsonify({"error": "No credits remaining"}), 402

    weights = {k: draft.get(k, 0) for k in [
        "Core Idea & Innovation", "Market & Commercial Opportunity",
        "Execution & Operations", "Business Model & Strategy",
        "Team & Organizational Health", "External Environment & Compliance",
        "Risk & Future Outlook"
    ]}
    try:
        res = ai_validate(draft["title"], draft["concept"], weights)
    except Exception as e:
        users_coll.update_one({"_id": uid}, {"$inc": {"creditQuota": 1, "noOfIdeas": -1}})
        return jsonify({"error": str(e)}), 500

    if res.get("error"):
        users_coll.update_one({"_id": uid}, {"$inc": {"creditQuota": 1, "noOfIdeas": -1}})
        return jsonify({"error": res["error"]}), 500

    score = float(res["overall_score"])
    status = "accepted" if score >= 3.5 else "improvise" if score >= 1.5 else "rejected"

    old_key = draft["pptFileKey"]
    new_key = old_key.replace("drafts/", "ideas/", 1)
    s3.copy_object(CopySource={"Bucket": BUCKET, "Key": old_key}, Bucket=BUCKET, Key=new_key)
    s3.delete_object(Bucket=BUCKET, Key=old_key)
    ppt_url = f"https://{BUCKET}.s3.amazonaws.com/{new_key}"

    idea_doc = {
        "_id": str(uuid.uuid4()),
        "userId": uid,
        "ideaName": draft["title"],
        "ideaConcept": draft["concept"],
        "overallScore": res["overall_score"],
        "status": status,
        "validationOutcome": res["validation_outcome"],
        "evaluatedData": res["evaluated_data"],
        "htmlReport": res["html_report"],
        "pptUrl": ppt_url,
        "weights": weights,
        "preset": draft.get("preset", "Balanced"),
        "domain": draft["domain"] if draft["domain"] != "Other" else draft.get("otherDomain", "Other"),
        "subDomain": draft.get("subDomain", ""),
        "cityOrVillage": draft.get("cityOrVillage", ""),
        "locality": draft.get("locality", ""),
        "trl": draft["trl"],
        "background": draft["background"],
        "coreTeamIds": [str(x) for x in draft.get("coreTeamIds", [])],
        "mentorId": str(draft["mentorId"]) if draft.get("mentorId") else None,
        "ttcCoordinatorId": user_doc.get("ttcCoordinatorId"),
        "collegeId": user_doc.get("collegeId"),
        "createdAt": dt.datetime.now(timezone.utc),
        "updatedAt": dt.datetime.now(timezone.utc),
        "isDeleted": False
    }
    ideas_coll.insert_one(idea_doc)
    drafts_coll.delete_one({"_id": draft_id})

    return jsonify({
        "success": True,
        "message": "Idea saved",
        "ideaId": idea_doc["_id"],
        "score": idea_doc["overallScore"],
        "pptUrl": ppt_url
    }), 201

# ---------- 8.  GET SINGLE DRAFT ----------
@app.route("/api/ideas/draft/<draft_id>", methods=["GET"])
@requires_role(["innovator"])
def get_draft(draft_id):
    uid = request.current_user["uid"]
    try:
        oid = parse_oid(draft_id, "draftId")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    draft = drafts_coll.find_one({"_id": oid, "ownerId": uid})
    if not draft:
        return jsonify({"error": "Draft not found"}), 404

    draft["draftId"] = str(draft.pop("_id"))  # give UI string key
    return jsonify(clean_doc(draft)), 200

# ---------- 9.  LIST USER IDEAS ----------
@app.route("/api/ideas/user/<uid>", methods=["GET"])
@requires_role(["innovator", "college_admin", "super_admin", "ttc_coordinator"])
def list_user_ideas(uid):
    role   = request.current_user["role"]
    caller = request.current_user["uid"]

    if role == "super_admin":
        query = {}
    elif role == "college_admin":
        innovator_ids = users_coll.distinct("_id", {"collegeId": caller})
        query = {"userId": {"$in": innovator_ids}}
    elif role == "ttc_coordinator":
        innovator_ids = users_coll.distinct("_id", {"createdBy": caller})
        query = {"userId": {"$in": innovator_ids}}
    else:  # innovator
        if caller != uid:
            return jsonify({"error": "Unauthorized"}), 403
        query = {"userId": uid}

    query["isDeleted"] = False

    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    skip  = (page - 1) * limit

    pipeline = [
        {"$match": query},
        {"$sort": {"createdAt": -1}},
        {"$skip": skip},
        {"$limit": limit},
        {"$project": {"htmlReport": 0, "evaluatedData": 0}},
        {"$lookup": {
            "from": "users",
            "localField": "userId",
            "foreignField": "_id",
            "as": "innovator"
        }},
        {"$unwind": "$innovator"},
        {"$lookup": {
            "from": "users",
            "localField": "innovator.createdBy",
            "foreignField": "_id",
            "as": "ttc"
        }},
        {"$unwind": {"path": "$ttc", "preserveNullAndEmptyArrays": True}},
        {"$project": {"innovator.password": 0, "ttc.password": 0}}
    ]

    docs = [clean_doc(d) for d in ideas_coll.aggregate(pipeline)]
    return jsonify(docs), 200

# ---------- 10.  GET SINGLE IDEA ----------
@app.route("/api/ideas/<idea_id>", methods=["GET"])
@requires_role(["innovator", "college_admin", "super_admin", "ttc_coordinator"])
def get_idea(idea_id):
    doc = ideas_coll.find_one({"_id": idea_id})
    if not doc:
        return jsonify({"error": "Not found"}), 404
    if doc["userId"] != request.current_user["uid"] and request.current_user["role"] not in {"super_admin", "college_admin", "ttc_coordinator"}:
        return jsonify({"error": "Unauthorized"}), 403
    if isinstance(doc.get("evaluatedData"), str):
        doc["evaluatedData"] = json.loads(doc["evaluatedData"])
    return jsonify(clean_doc(doc)), 200

# Credit requests
@app.route("/api/colleges/<cid>/credit-requests", methods=["POST"])
@requires_role(["college_admin"])
def credit_request(cid):
    body = request.get_json(force=True)
    amt, reason = body.get("amount"), body.get("reason")
    if not amt or not reason:
        return jsonify({"error": "amount & reason required"}), 400
    rid = str(uuid.uuid4())
    credit_coll(cid).insert_one({
        "_id": rid, "collegeId": cid, "requestedBy": request.current_user["uid"],
        "amount": amt, "reason": reason, "status": "pending",
        "createdAt": datetime.now(timezone.utc)
    })
    return jsonify({"message": "Credit request submitted", "requestId": rid}), 201

@app.route("/api/colleges/<cid>/credit-requests", methods=["GET"])
@requires_role(["college_admin", "super_admin"])
def list_credit_requests(cid):
    cursor = credit_coll(cid).find().sort("createdAt", -1)
    return jsonify(list(cursor)), 200

@app.route("/api/colleges/<cid>/credit-requests/<rid>/status", methods=["PUT"])
@requires_role(["super_admin"])
def update_credit_status(cid, rid):
    status = request.json.get("status")
    if status not in {"approved", "rejected"}:
        return jsonify({"error": "Invalid status"}), 400
    res = credit_coll(cid).update_one(
        {"_id": rid}, {"$set": {"status": status, "updatedAt": datetime.now(timezone.utc)}}
    )
    if res.matched_count == 0:
        return jsonify({"error": "Request not found"}), 404
    return jsonify({"message": f"Status updated to {status}"}), 200

# Training programs
@app.route("/api/ttc/programs", methods=["POST"])
@requires_role(["ttc_coordinator", "super_admin"])
def create_program():
    body = request.get_json(force=True)
    name, desc, date, cap = body.get("name"), body.get("description"), body.get("date"), body.get("capacity")
    if not all([name, desc, date, cap]):
        return jsonify({"error": "All fields required"}), 400
    pid = str(uuid.uuid4())
    program_coll.insert_one({
        "_id": pid, "name": name, "description": desc, "date": date,
        "capacity": int(cap), "registeredCount": 0,
        "createdBy": request.current_user["uid"], "createdAt": datetime.now(timezone.utc)
    })
    return jsonify({"message": "Program created", "programId": pid}), 201

@app.route("/api/ttc/programs", methods=["GET"])
@requires_role(["ttc_coordinator", "super_admin", "innovator", "college_admin"])
def list_programs():
    cursor = program_coll.find().sort("date", 1)
    return jsonify(list(cursor)), 200

@app.route("/api/ttc/programs/<pid>/register", methods=["POST"])
@requires_role(["innovator", "college_admin"])
def register_program(pid):
    user_id = request.current_user["uid"]
    res = program_coll.update_one(
        {"_id": pid, "registeredCount": {"$lt": "$capacity"}},
        {
            "$inc": {"registeredCount": 1},
            "$addToSet": {"registrations": user_id}
        }
    )
    if res.modified_count == 0:
        return jsonify({"error": "Program full or not found"}), 400
    return jsonify({"message": "Registered"}), 200


# Analytics apis
# ------------------------------------------------------------
# College Admin : domain-wise idea counts
# ------------------------------------------------------------
# ------------------------------------------------------------
# College Admin : domain-wise idea counts (fast, index-friendly)
# ------------------------------------------------------------
@app.route("/api/analytics/college/domain-trend/<collegeId>", methods=["GET"])
@requires_role(["college_admin"])
def college_domain_trend(collegeId):
    caller_id = request.current_user["uid"]
    print("caller_id:", caller_id)
    print("collegeId param:", collegeId)

    pipeline = [
        {"$match": {
            "collegeId": collegeId,
            "isDeleted": {"$ne": True}
        }},
        {"$group": {
            "_id": "$domain",
            "ideas": {"$sum": 1}
        }},
        {"$sort": {"ideas": -1}},
        {"$project": {
            "_id": 0,
            "name": "$_id",
            "ideas": 1
        }}
    ]
    data = list(ideas_coll.aggregate(pipeline))

    return jsonify({"success": True, "data": data}), 200

#-------------------------------------------------------------------------
#   domain-wise idea counts
#-------------------------------------------------------------------------
@app.route("/api/analytics/domain-trend", methods=["GET"])
@requires_auth
def domain_trend():
    
    caller = request.current_user
    caller_role = caller["role"]
    caller_id   = caller["uid"]

    # Build the filter
    if caller_role == "super_admin":
        # super-admin can see everything
        match_stage = {"isDeleted": {"$ne": True}}
    elif caller_role == "college_admin":
        # use the caller’s own collegeId
        match_stage = {
            "collegeId": caller_id,
            "isDeleted": {"$ne": True}
        }
    elif caller_role == "ttc_coordinator":
        # only ideas created by users whose createdBy == caller_id
        innovator_ids = users_coll.distinct("_id", {"createdBy": caller_id})
        match_stage = {
            "userId": {"$in": innovator_ids},
            "isDeleted": {"$ne": True}
        }
    elif caller_role == "innovator":
        # only caller’s own ideas
        match_stage = {
            "userId": caller_id,
            "isDeleted": {"$ne": True}
        }
    else:
        return jsonify({"error": "Unknown role"}), 403

    pipeline = [
        {"$match": match_stage},
        {"$group": {
            "_id": "$domain",
            "ideas": {"$sum": 1}
        }},
        {"$sort": {"ideas": -1}},
        {"$project": {
            "_id": 0,
            "name": "$_id",
            "ideas": 1
        }}
    ]
    data = list(ideas_coll.aggregate(pipeline))
    return jsonify({"success": True, "data": data}), 200

#-------------------------------------------------------------------------
#   Top performing innovators
#-------------------------------------------------------------------------
@app.route("/api/analytics/top-innovators", methods=["GET"])
@requires_role(["ttc_coordinator", "college_admin"])
def top_innovators():
    caller_id = request.current_user["uid"]
    caller_role = request.current_user["role"]

    # 1. Determine which innovators belong to this caller
    if caller_role == "ttc_coordinator":
        innovator_ids = users_coll.distinct("_id", {"createdBy": caller_id})
    elif caller_role == "college_admin":
        innovator_ids = users_coll.distinct("_id", {"collegeId": caller_id})
    else:
        return jsonify([]), 200  # defensive

    if not innovator_ids:
        return jsonify([]), 200

    # 2. Aggregate idea scores per innovator
    pipeline = [
        {"$match": {
            "userId": {"$in": innovator_ids},
            "isDeleted": {"$ne": True}
        }},
        {"$group": {
            "_id": "$userId",
            "avgScore": {"$avg": "$overallScore"},
            "ideaCount": {"$sum": 1}
        }},
        {"$sort": {"avgScore": -1}},
        {"$limit": 5},                     # top 5
        {"$lookup": {
            "from": "users",
            "localField": "_id",
            "foreignField": "_id",
            "as": "user"
        }},
        {"$unwind": "$user"},
        {"$project": {
            "_id": 0,
            "name": "$user.name",
            "score": "$avgScore"
        }}
    ]

    data = list(ideas_coll.aggregate(pipeline))
    return jsonify(data), 200

#---------------------------------------------------------------------------------
#   Innovator engagemnet
#---------------------------------------------------------------------------------

@app.route("/api/analytics/innovator-engagement", methods=["GET"])
@requires_role(["ttc_coordinator", "college_admin"])
def innovator_engagement():
    caller = request.current_user
    role   = caller["role"]
    caller_id = caller["uid"]

    # Build the innovator filter once
    if role == "ttc_coordinator":
        innovator_ids = users_coll.distinct("_id", {"createdBy": caller_id, "role": "innovator"})
    else:  # college_admin
        innovator_ids = users_coll.distinct("_id", {"collegeId": caller_id, "role": "innovator"})

    active_cnt = users_coll.count_documents({
        "_id": {"$in": innovator_ids},
        "isActive": True,
        "isDeleted": {"$ne": True}
    })
    invited_cnt = len(innovator_ids) - active_cnt

    data = [
        {"name": "Active Innovators",   "value": active_cnt,  "fill": "hsl(var(--chart-1))"},
        {"name": "Invited Innovators",  "value": invited_cnt, "fill": "hsl(var(--chart-5))"}
    ]
    return jsonify({"success": True, "data": data}), 200


#---------------------------------------------------------------------------------
#   Idea Quality
#---------------------------------------------------------------------------------
@app.route("/api/analytics/idea-quality-trend", methods=["GET"])
@requires_role(["ttc_coordinator", "college_admin"])
def idea_quality_trend():
    caller = request.current_user
    role   = caller["role"]
    caller_id = caller["uid"]

    if role == "ttc_coordinator":
        innovator_ids = users_coll.distinct("_id", {"createdBy": caller_id, "role": "innovator"})
    else:
        innovator_ids = users_coll.distinct("_id", {"collegeId": caller_id, "role": "innovator"})

    pipeline = [
        {"$match": {"userId": {"$in": innovator_ids}, "isDeleted": {"$ne": True}}},
        {"$addFields": {"month": {"$dateToString": {"format": "%b", "date": "$createdAt"}}}},
        {"$group": {"_id": "$month", "quality": {"$avg": "$overallScore"}}},
        {"$sort": {"_id": 1}}
    ]
    raw = list(ideas_coll.aggregate(pipeline))

    month_order = ["Jan","Feb","Mar","Apr","May","Jun"]
    mapped = {m["_id"]: round(m["quality"], 2) for m in raw}
    data = [{"month": m, "quality": mapped.get(m, 0)} for m in month_order]

    return jsonify({"success": True, "data": data}), 200


#---------------------------------------------------------------------------------
#   category-success
#---------------------------------------------------------------------------------
@app.route("/api/analytics/category-success", methods=["GET"])
@requires_role(["ttc_coordinator", "college_admin"])
def category_success():
    caller = request.current_user
    role   = caller["role"]
    caller_id = caller["uid"]

    if role == "ttc_coordinator":
        innovator_ids = users_coll.distinct("_id", {"createdBy": caller_id, "role": "innovator"})
    else:
        innovator_ids = users_coll.distinct("_id", {"collegeId": caller_id, "role": "innovator"})

    pipeline = [
        {"$match": {"userId": {"$in": innovator_ids}, "isDeleted": {"$ne": True}}},
        {"$group": {
            "_id": "$domain",
            "approved":  {"$sum": {"$cond": [{"$gte": ["$overallScore", 80]}, 1, 0]}},
            "moderate":  {"$sum": {"$cond": [{"$and": [{"$gte": ["$overallScore", 50]}, {"$lt": ["$overallScore", 80]}]}, 1, 0]}},
            "rejected":  {"$sum": {"$cond": [{"$lt": ["$overallScore", 50]}, 1, 0]}}
        }},
        {"$project": {"_id": 0, "category": "$_id", "approved": 1, "moderate": 1, "rejected": 1}}
    ]
    data = list(ideas_coll.aggregate(pipeline))
    return jsonify({"success": True, "data": data}), 200


#---------------------------------------------------------------------------------
#   Rejection reasons
#---------------------------------------------------------------------------------

@app.route("/api/analytics/rejection-reasons", methods=["GET"])
@requires_role(["ttc_coordinator", "college_admin"])
def rejection_reasons():
    caller = request.current_user
    role   = caller["role"]
    caller_id = caller["uid"]

    if role == "ttc_coordinator":
        innovator_ids = users_coll.distinct("_id", {"createdBy": caller_id, "role": "innovator"})
    else:
        innovator_ids = users_coll.distinct("_id", {"collegeId": caller_id, "role": "innovator"})

    pipeline = [
        {"$match": {
            "userId": {"$in": innovator_ids},
            "overallScore": {"$lt": 50},
            "isDeleted": {"$ne": True}
        }},
        {"$unwind": "$evaluatedData"},
        {"$match": {"evaluatedData.score": {"$lt": 50}}},
        {"$group": {"_id": "$evaluatedData.criterion", "value": {"$sum": 1}}},
        {"$sort": {"value": -1}},
        {"$limit": 4},
        {"$project": {
            "_id": 0,
            "name": "$_id",
            "value": 1,
            "fill": {
                "$switch": {
                    "branches": [
                        {"case": {"$eq": ["$_id", "Low Market Need"]},       "then": "hsl(var(--color-rejected))"},
                        {"case": {"$eq": ["$_id", "Technical Feasibility"]}, "then": "hsl(var(--chart-3))"},
                        {"case": {"$eq": ["$_id", "Weak Business Model"]},  "then": "hsl(var(--chart-5))"},
                        {"case": {"$eq": ["$_id", "Poor Team Fit"]},        "then": "hsl(var(--muted))"}
                    ],
                    "default": "hsl(var(--muted))"
                }
            }
        }}
    ]
    data = list(ideas_coll.aggregate(pipeline))
    if not data:
        data = [{"name": "No rejections yet", "value": 1, "fill": "hsl(var(--muted))"}]
    return jsonify({"success": True, "data": data}), 200



# ---------------------------------------------------------------
# 8.  Health & run
# ---------------------------------------------------------------
@app.route("/")
def health(): return jsonify({"status": "ok"})

def create_app(): return app

if __name__ == "__main__":
    app.run(debug=True)