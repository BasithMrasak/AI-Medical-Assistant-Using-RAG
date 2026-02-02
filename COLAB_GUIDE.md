# Google Colab Integration Guide

This guide explains how to push your Google Colab notebook code into this repository.

## Overview

Google Colab is an excellent platform for developing and testing AI/ML projects. This guide covers different methods to integrate your Colab work with this GitHub repository.

## Prerequisites

- A Google account with access to Google Colab
- A GitHub account with access to this repository
- Basic familiarity with Git and GitHub

## Method 1: Download and Upload (Easiest)

This is the simplest method for beginners.

### Steps:

1. **Save your work in Google Colab**
   - Click `File` → `Save` to ensure your changes are saved

2. **Download the notebook**
   - Click `File` → `Download` → `Download .ipynb`
   - The notebook will be downloaded to your local machine

3. **Upload to GitHub**
   - Go to this repository on GitHub
   - Click `Add file` → `Upload files`
   - Drag and drop your `.ipynb` file
   - Add a commit message describing your changes
   - Click `Commit changes`

## Method 2: Using Git Commands in Colab

This method allows you to work directly with Git from within Google Colab.

### Steps:

1. **Mount Google Drive (optional, for persistent storage)**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Configure Git**
   ```bash
   !git config --global user.email "your-email@example.com"
   !git config --global user.name "Your Name"
   ```

3. **Clone the repository**
   ```bash
   !git clone https://github.com/BasithMrasak/AI-Medical-Assistant-Using-RAG.git
   %cd AI-Medical-Assistant-Using-RAG
   ```

4. **Create or update your notebook**
   - Work on your notebook in Colab
   - Save it: `File` → `Download` → `Download .ipynb`
   - Upload the notebook to the Colab file browser
   - Move it to the cloned repository folder

5. **Commit and push your changes**
   ```bash
   !git add *.ipynb
   !git commit -m "Add/Update notebook from Google Colab"
   !git push
   ```

   **Note:** For authentication, you'll need to use a Personal Access Token:
   - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate a new token with `repo` scope
   - Use it as your password when pushing

## Method 3: Using GitHub Integration (Recommended)

Google Colab has built-in integration with GitHub that makes the process smoother.

### Steps:

1. **Open notebook from GitHub**
   - In Google Colab, click `File` → `Open notebook`
   - Select the `GitHub` tab
   - Enter the repository URL: `BasithMrasak/AI-Medical-Assistant-Using-RAG`
   - Select the notebook you want to edit (or create a new one)

2. **Make your changes**
   - Edit and run your code in Colab
   - Test thoroughly

3. **Save back to GitHub**
   - Click `File` → `Save a copy in GitHub`
   - Select the repository: `BasithMrasak/AI-Medical-Assistant-Using-RAG`
   - Choose the branch (usually `main` or create a new branch)
   - Add a commit message
   - Click `OK`

### Advantages:
- No need to download/upload files manually
- Direct integration with version control
- Can create branches and pull requests easily

## Method 4: Using Google Drive + Git Sync

For more advanced users who want automatic syncing.

### Steps:

1. **Mount Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Set up repository in Drive**
   ```bash
   %cd /content/drive/MyDrive
   !git clone https://github.com/BasithMrasak/AI-Medical-Assistant-Using-RAG.git
   %cd AI-Medical-Assistant-Using-RAG
   ```

3. **Work on your notebook**
   - Create/edit notebooks in this directory
   - All changes are automatically saved to Google Drive

4. **Commit and push when ready**
   ```bash
   !git add .
   !git status
   !git commit -m "Your commit message"
   !git push
   ```

## Best Practices

1. **Organize Your Notebooks**
   - Create a `notebooks/` directory for Jupyter notebooks
   - Use descriptive names for your notebooks (e.g., `01_data_preprocessing.ipynb`, `02_model_training.ipynb`)

2. **Clear Outputs Before Committing**
   - Click `Edit` → `Clear all outputs` before saving
   - This keeps the repository clean and reduces file size

3. **Use Requirements.txt**
   - Document all dependencies in `requirements.txt`
   - This helps others reproduce your environment

4. **Add Documentation**
   - Include markdown cells explaining your code
   - Add a README in the notebooks folder if needed

5. **Version Control**
   - Make frequent, small commits with clear messages
   - Use branches for experimental features
   - Create pull requests for review before merging to main

6. **Test Your Code**
   - Ensure your notebook runs from top to bottom without errors
   - Test with `Runtime` → `Restart and run all`

## Troubleshooting

### Authentication Issues
- Use Personal Access Tokens instead of passwords
- Generate tokens at: https://github.com/settings/tokens
- Store tokens securely (don't commit them to the repository)

### Large Files
- Avoid committing large data files or model binaries
- Use `.gitignore` to exclude them
- Consider using Git LFS for large files or external storage

### Merge Conflicts
- Pull latest changes before starting work: `!git pull`
- If conflicts occur, resolve them manually or use GitHub's web interface

## Additional Resources

- [Google Colab Documentation](https://colab.research.google.com/)
- [GitHub Documentation](https://docs.github.com/)
- [Git Tutorial](https://git-scm.com/docs/gittutorial)

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Search for similar issues on GitHub
3. Open an issue in this repository with details about your problem
