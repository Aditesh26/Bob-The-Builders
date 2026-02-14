from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

try:
    password = "testpassword123"
    print(f"Hashing password: '{password}' (len={len(password)})")
    hashed = pwd_context.hash(password)
    print(f"Hashed: {hashed}")
    
    print("Verifying...")
    valid = pwd_context.verify(password, hashed)
    print(f"Valid: {valid}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
