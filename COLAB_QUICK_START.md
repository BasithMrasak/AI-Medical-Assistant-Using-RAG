# Quick Start Guide: Pushing from Google Colab to GitHub

This is a quick reference for pushing your notebook work from Google Colab to this repository.

## 🚀 Quick Method (Recommended for Beginners)

### Step 1: Open Your Notebook in Colab

### Step 2: Save to GitHub Directly
1. Click `File` > `Save a copy in GitHub`
2. Authorize Colab to access GitHub (first time only)
3. Select repository: `BasithMrasak/AI-Medical-Assistant-Using-RAG`
4. Choose/create a branch (e.g., `feature/my-changes`)
5. Add commit message
6. Click **OK**

✅ Done! Your notebook is now in the repository.

### Step 3: Create a Pull Request
1. Go to https://github.com/BasithMrasak/AI-Medical-Assistant-Using-RAG
2. You'll see a prompt to create a Pull Request
3. Click "Compare & pull request"
4. Fill in the description and submit

---

## 💻 Advanced Method (Using Git Commands)

For more control, use Git commands in Colab cells:

### Initial Setup (Do Once)
```python
# Configure Git
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Clone repository
!git clone https://github.com/BasithMrasak/AI-Medical-Assistant-Using-RAG.git
%cd AI-Medical-Assistant-Using-RAG
```

### Regular Workflow
```python
# Create a branch
!git checkout -b feature/my-feature

# After making changes to notebooks...

# Add and commit
!git add .
!git commit -m "Description of changes"

# Push (requires authentication)
!git push origin feature/my-feature
```

### Authentication
When prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Your Personal Access Token (not your GitHub password!)

To create a token:
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select `repo` scope
4. Copy the token and use it as your password

---

## 📁 Using Google Drive (Persist Across Sessions)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to Drive and clone
%cd /content/drive/MyDrive/
!git clone https://github.com/BasithMrasak/AI-Medical-Assistant-Using-RAG.git
%cd AI-Medical-Assistant-Using-RAG

# Now work normally - changes persist!
```

---

## ❓ Troubleshooting

**Problem**: "Authentication failed"
- **Solution**: Use a Personal Access Token (see above) instead of your password

**Problem**: "Permission denied"
- **Solution**: Make sure you have write access to the repository

**Problem**: "Large files won't push"
- **Solution**: GitHub has a 100MB limit. Don't commit large model files directly.

**Problem**: "Session disconnected, lost my work"
- **Solution**: Use the Google Drive method or commit frequently

---

## 📚 Need More Help?

See the full [CONTRIBUTING.md](CONTRIBUTING.md) guide for:
- Detailed explanations of each method
- Code style guidelines
- Best practices
- More troubleshooting tips

Or check the [example notebook](example_medical_assistant.ipynb) to see the project in action!
