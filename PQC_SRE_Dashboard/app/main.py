# # 


# # --

# # File: app/main.py
# # Facial Recognition + Post-Quantum Cryptography (Kyber via oqs-python)

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import base64, numpy as np
# from typing import List, Optional
# import oqs
# import os
# import uvicorn
# from cryptography.hazmat.primitives.ciphers.aead import AESGCM
# import secrets

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class EncryptedDescriptor(BaseModel):
#     username: str
#     encrypted_descriptor: str  # base64
#     ciphertext: str  # base64 (Kyber ciphertext)

# class Ticket(BaseModel):
#     id: int
#     title: str
#     description: str
#     status: str
#     application: Optional[str] = "monolith"

# registered_faces = {
#     "user123": {
#         "face_descriptor": np.random.rand(128),
#         "kyber_private_key": None
#     }
# }

# app_health_status = {
#     "status": "Healthy",
#     "uptime": "99.97%",
#     "latency_ms": 95,
#     "error_rate": 0.002,
#     "slo": "99.9% uptime",
#     "ski": "<200ms latency",
#     "performance_28_days": [
#         {"date": "2025-07-01", "uptime": 99.9, "latency": 98},
#         {"date": "2025-07-02", "uptime": 99.95, "latency": 91},
#     ]
# }

# tickets = []

# @app.get("/register/kyber/{username}")
# def register_kyber(username: str):
#     kem = oqs.KeyEncapsulation("Kyber512")
#     public_key = kem.generate_keypair()
#     registered_faces[username] = {
#         "face_descriptor": np.random.rand(128),
#         "kyber_private_key": kem.export_secret_key()
#     }
#     return {"public_key": base64.b64encode(public_key).decode()}

# @app.post("/login")
# def login(data: EncryptedDescriptor):
#     user = registered_faces.get(data.username)
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     ciphertext = base64.b64decode(data.ciphertext)
#     encrypted_descriptor = base64.b64decode(data.encrypted_descriptor)

#     # Recreate KEM object and load the user's private key
#     kem = oqs.KeyEncapsulation("Kyber512")
#     kem.import_secret_key(user["kyber_private_key"])
#     shared_secret = kem.decap_secret(ciphertext)

#     # Use shared_secret to decrypt facial descriptor using AES-GCM
#     nonce = encrypted_descriptor[:12]
#     ciphertext_body = encrypted_descriptor[12:]
#     aesgcm = AESGCM(shared_secret[:16])
#     decrypted_descriptor_bytes = aesgcm.decrypt(nonce, ciphertext_body, None)
#     decrypted_descriptor = np.frombuffer(decrypted_descriptor_bytes, dtype=np.float32)

#     sim = np.dot(user["face_descriptor"], decrypted_descriptor) / (
#         np.linalg.norm(user["face_descriptor"]) * np.linalg.norm(decrypted_descriptor))

#     return {"authenticated": sim > 0.85, "similarity": sim}

# @app.get("/dashboard/health")
# def get_dashboard_health():
#     return app_health_status

# @app.get("/tickets", response_model=List[Ticket])
# def get_all_tickets():
#     return tickets

# @app.post("/tickets", response_model=Ticket)
# def create_ticket(ticket: Ticket):
#     tickets.append(ticket)
#     return ticket

# @app.put("/tickets/{ticket_id}", response_model=Ticket)
# def update_ticket(ticket_id: int, updated_ticket: Ticket):
#     for i, t in enumerate(tickets):
#         if t.id == ticket_id:
#             tickets[i] = updated_ticket
#             return updated_ticket
#     raise HTTPException(status_code=404, detail="Ticket not found")

# @app.delete("/tickets/{ticket_id}")
# def delete_ticket(ticket_id: int):
#     global tickets
#     tickets = [t for t in tickets if t.id != ticket_id]
#     return {"message": f"Ticket {ticket_id} deleted"}

# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)



# next


# File: app/main.py
# Facial Recognition + Post-Quantum Cryptography (Kyber via oqs-python)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import base64, numpy as np
from typing import List, Optional
import oqs
import os
import uvicorn
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join("static", "index.html"))

class EncryptedDescriptor(BaseModel):
    username: str
    encrypted_descriptor: str  # base64
    ciphertext: str  # base64 (Kyber ciphertext)

class Ticket(BaseModel):
    id: int
    title: str
    description: str
    status: str
    application: Optional[str] = "monolith"

registered_faces = {
    "user123": {
        "face_descriptor": np.random.rand(128),
        "kyber_private_key": None
    }
}

app_health_status = {
    "status": "Healthy",
    "uptime": "99.97%",
    "latency_ms": 95,
    "error_rate": 0.002,
    "slo": "99.9% uptime",
    "ski": "<200ms latency",
    "performance_28_days": [
        {"date": "2025-07-01", "uptime": 99.9, "latency": 98},
        {"date": "2025-07-02", "uptime": 99.95, "latency": 91},
    ]
}

tickets = []

@app.get("/register/kyber/{username}")
def register_kyber(username: str):
    kem = oqs.KeyEncapsulation("Kyber512")
    public_key = kem.generate_keypair()
    registered_faces[username] = {
        "face_descriptor": np.random.rand(128),
        "kyber_private_key": kem.export_secret_key()
    }
    return {"public_key": base64.b64encode(public_key).decode()}

@app.post("/login")
def login(data: EncryptedDescriptor):
    user = registered_faces.get(data.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    ciphertext = base64.b64decode(data.ciphertext)
    encrypted_descriptor = base64.b64decode(data.encrypted_descriptor)

    # Recreate KEM object and load the user's private key
    kem = oqs.KeyEncapsulation("Kyber512")
    kem.import_secret_key(user["kyber_private_key"])
    shared_secret = kem.decap_secret(ciphertext)

    # Use shared_secret to decrypt facial descriptor using AES-GCM
    nonce = encrypted_descriptor[:12]
    ciphertext_body = encrypted_descriptor[12:]
    aesgcm = AESGCM(shared_secret[:16])
    decrypted_descriptor_bytes = aesgcm.decrypt(nonce, ciphertext_body, None)
    decrypted_descriptor = np.frombuffer(decrypted_descriptor_bytes, dtype=np.float32)

    sim = np.dot(user["face_descriptor"], decrypted_descriptor) / (
        np.linalg.norm(user["face_descriptor"]) * np.linalg.norm(decrypted_descriptor))

    return {"authenticated": sim > 0.85, "similarity": sim}

@app.get("/dashboard/health")
def get_dashboard_health():
    return app_health_status

@app.get("/tickets", response_model=List[Ticket])
def get_all_tickets():
    return tickets

@app.post("/tickets", response_model=Ticket)
def create_ticket(ticket: Ticket):
    tickets.append(ticket)
    return ticket

@app.put("/tickets/{ticket_id}", response_model=Ticket)
def update_ticket(ticket_id: int, updated_ticket: Ticket):
    for i, t in enumerate(tickets):
        if t.id == ticket_id:
            tickets[i] = updated_ticket
            return updated_ticket
    raise HTTPException(status_code=404, detail="Ticket not found")

@app.delete("/tickets/{ticket_id}")
def delete_ticket(ticket_id: int):
    global tickets
    tickets = [t for t in tickets if t.id != ticket_id]
    return {"message": f"Ticket {ticket_id} deleted"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
