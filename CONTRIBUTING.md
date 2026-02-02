# Contributing to AI Medical Assistant Using RAG

Thank you for your interest in contributing to this project! This guide will help you set up your development environment and contribute code, especially if you're working in Google Colab.

## Table of Contents
- [Working with Google Colab](#working-with-google-colab)
  - [Method 1: Using Git Commands in Colab](#method-1-using-git-commands-in-colab)
  - [Method 2: Using Google Drive Integration](#method-2-using-google-drive-integration)
  - [Method 3: Using Colab's GitHub Integration](#method-3-using-colabs-github-integration)
- [Local Development](#local-development)
- [Code Style and Best Practices](#code-style-and-best-practices)
- [Troubleshooting](#troubleshooting)

## Working with Google Colab

Google Colab is a great platform for developing and testing this AI Medical Assistant. Here are three methods to push your notebook code to this repository:

### Method 1: Using Git Commands in Colab

This is the recommended method for full control over your commits.

#### Step 1: Set up Git credentials

First, configure Git with your credentials in a Colab cell:

```python
# Run this in a Colab cell
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"
```

#### Step 2: Clone the repository

```python
# Clone the repository
!git clone https://github.com/BasithMrasak/AI-Medical-Assistant-Using-RAG.git
%cd AI-Medical-Assistant-Using-RAG
```

#### Step 3: Create a new branch for your changes

```python
# Create and switch to a new branch
!git checkout -b feature/your-feature-name
```

#### Step 4: Work on your notebook

- Create or modify notebooks in the cloned directory
- Save your work regularly using `File > Save` in Colab

#### Step 5: Commit and push your changes

```python
# Add your changes
!git add .

# Commit your changes
!git commit -m "Add: Description of your changes"

# Push to GitHub (you'll need to authenticate)
!git push origin feature/your-feature-name
```

#### Authentication Options:

**Option A: Using Personal Access Token (Recommended)**

1. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Generate a new token with `repo` permissions
3. Use the token when prompted for password:

```python
# When pushing, use your GitHub username and token
# Username: your-github-username
# Password: your-personal-access-token
```

**Option B: Using GitHub CLI**

```python
# Install GitHub CLI
!curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
!echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
!sudo apt update
!sudo apt install gh -y

# Authenticate
!gh auth login
```

### Method 2: Using Google Drive Integration

This method allows you to work with files synchronized through Google Drive.

#### Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Step 2: Clone the repository to Google Drive

```python
# Navigate to your Drive
%cd /content/drive/MyDrive/

# Clone the repository
!git clone https://github.com/BasithMrasak/AI-Medical-Assistant-Using-RAG.git
%cd AI-Medical-Assistant-Using-RAG
```

#### Step 3: Configure Git and work normally

Your changes will persist across Colab sessions since they're stored in Google Drive.

```python
# Configure Git
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Create a branch
!git checkout -b feature/your-feature-name

# After making changes, commit and push
!git add .
!git commit -m "Your commit message"
!git push origin feature/your-feature-name
```

### Method 3: Using Colab's GitHub Integration

Colab provides built-in GitHub integration for saving notebooks.

#### Step 1: Save to GitHub directly

1. In your Colab notebook, go to `File > Save a copy in GitHub`
2. Authorize Colab to access your GitHub account
3. Select the repository: `BasithMrasak/AI-Medical-Assistant-Using-RAG`
4. Choose the branch or create a new one
5. Add a commit message
6. Click OK

#### Step 2: Create a Pull Request

After pushing your notebook, go to GitHub and create a pull request from your branch to main.

## Local Development

If you prefer to work locally instead of Google Colab:

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Setup

1. Clone the repository:
```bash
git clone https://github.com/BasithMrasak/AI-Medical-Assistant-Using-RAG.git
cd AI-Medical-Assistant-Using-RAG
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Jupyter (if not already installed):
```bash
pip install jupyter notebook
```

5. Start working:
```bash
jupyter notebook
```

## Code Style and Best Practices

### Jupyter Notebooks

- Clear all outputs before committing (optional, but keeps diffs clean):
  ```python
  # In a notebook cell
  from IPython.display import clear_output
  clear_output()
  ```
  
- Add markdown cells to explain your code and methodology
- Use meaningful variable names
- Keep cells focused on single tasks
- Add comments for complex logic

### Python Code

- Follow PEP 8 style guidelines
- Use descriptive variable and function names
- Add docstrings to functions and classes
- Keep functions small and focused

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove)
- Keep the first line under 50 characters
- Add details in the commit body if needed

Example:
```
Add medical query classification model

- Implemented transformer-based classifier
- Added training script with evaluation metrics
- Updated requirements.txt with new dependencies
```

### Creating Pull Requests

1. Create a new branch for your feature
2. Make your changes
3. Test your changes thoroughly
4. Push to your fork or branch
5. Create a pull request with a clear description

## Troubleshooting

### Issue: Authentication fails when pushing

**Solution**: Use a Personal Access Token instead of your password:
1. Create a token at https://github.com/settings/tokens
2. Use it as your password when Git prompts you

### Issue: Colab session disconnects and I lose my work

**Solution**: 
- Use Method 2 (Google Drive Integration) to persist your work
- Regularly commit and push your changes
- Save notebooks frequently

### Issue: Large files won't push to GitHub

**Solution**: 
- GitHub has a 100MB file size limit
- Use Git LFS for large files:
```python
!git lfs install
!git lfs track "*.bin"  # Track large model files
!git add .gitattributes
```

### Issue: Merge conflicts

**Solution**:
```python
# Pull latest changes
!git pull origin main

# If conflicts occur, edit the files to resolve them
# Then commit the resolution
!git add .
!git commit -m "Resolve merge conflicts"
!git push origin your-branch-name
```

### Issue: Need to download notebook from Colab

**Solution**:
- File > Download > Download .ipynb
- Or use the Git methods above to push directly

### Issue: Want to work on someone else's pull request

**Solution**:
```python
# Fetch all branches
!git fetch origin

# Checkout the PR branch
!git checkout -b branch-name origin/branch-name
```

## Questions or Need Help?

If you have questions or run into issues:
1. Check existing issues on GitHub
2. Create a new issue with a detailed description
3. Join our discussions (if applicable)

Thank you for contributing to the AI Medical Assistant project! 🚀
