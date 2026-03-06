# ─────────────────────────────────────────────
# STEP 1: Choose the base image
# Think of this as: "Start with a clean computer
# that already has Python 3.12 installed"
# ─────────────────────────────────────────────
FROM python:3.12-slim

# ─────────────────────────────────────────────
# STEP 2: Set the working directory INSIDE
# the container — all commands run from here
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
# STEP 3: Copy requirements FIRST
# Why first? Docker caches each step — if
# requirements haven't changed, it skips
# reinstalling everything! Huge time saver!
# ─────────────────────────────────────────────
COPY requirements.txt .

# ─────────────────────────────────────────────
# STEP 4: Install dependencies
# --no-cache-dir keeps the image size small
# ─────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# STEP 5: Copy the rest of the app
# ─────────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────────
# STEP 6: Tell Docker which port the app uses
# This is documentation — doesn't actually
# open the port, just declares intent!
# ─────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────
# STEP 7: The command to run when the container
# starts — notice NO reload=True in production!
# ─────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]