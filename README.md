# Resume-analyzer
# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get:

- An overall **ATS score (0-100)** with a 5-category breakdown
- Matched and **missing keywords**
- Strengths, weaknesses and **prioritised improvements**
- Suggested **bullet-point rewrites**
- Instant rule-based checks (email, phone, sections, metrics)
- A downloadable Markdown report

Built with [Streamlit](https://streamlit.io) and Google Gemini Flash.

## How the score works

Gemini rates the resume from 0-100 in five categories. The overall score is computed in code
(not by the model) so it is consistent:

| Category | Weight |
|---|---|
| Keyword match | 30% |
| Content & impact | 25% |
| ATS-friendly formatting | 15% |
| Structure & sections | 15% |
| Readability & grammar | 15% |

> This is an estimate. Real ATS software differs between employers, and formatting is judged from
> extracted text (the app cannot see visual layout). Paste a job description for the best keyword score.

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either paste it into the sidebar
or create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
# Optional: override the model
# GEMINI_MODEL = "gemini-3.6-flash"
```

Start the app:

```bash
streamlit run app.py
```

## Change the Gemini model

Gemini model names change often. The default is set in `DEFAULT_MODEL` in `app.py`.
You can override it without editing code via the sidebar field or the `GEMINI_MODEL` secret.
Current names: https://ai.google.dev/gemini-api/docs/models

## Deploy on Streamlit Community Cloud

1. Push this project to a public or private GitHub repository (`app.py` and `requirements.txt` at the repo root).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** -> **Deploy a public app from GitHub**.
4. Choose your repository, branch `main`, and main file path `app.py`.
5. Open **Advanced settings** -> **Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
6. Click **Deploy**. Your app gets a public `*.streamlit.app` URL.

## Privacy

Resume text is sent to the Gemini API for analysis. The app does not store files or results.
Never commit your API key; `.gitignore` already excludes `.streamlit/secrets.toml`.

## Project files

```
app.py            # Streamlit app
requirements.txt  # Dependencies
README.md         # This file
.gitignore        # Keeps secrets out of Git
```
