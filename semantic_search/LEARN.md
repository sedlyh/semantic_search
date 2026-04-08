# LEARN.md — Semantic Search for Florida Real Estate (Complete Study Guide)

> This document walks you through **every file, every line, and every concept** in this project.
> It is written for someone who knows what variables, functions, imports, loops, and classes are — but is encountering most of these libraries and patterns for the first time.
> **Note:** The Streamlit UI (`search_app.py`) and root `Dockerfile` were removed. Use **FastAPI** + **Next.js** only; deploy the API using any Python host you choose, as described in `README.md`.

> Read it top-to-bottom the first time; use the table of contents to jump around later.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites & Environment](#2-prerequisites--environment)
3. [Dependencies — Deep Dive](#3-dependencies--deep-dive)
   - [Python Dependencies](#31-python-dependencies-requirementstxt)
   - [JavaScript/Next.js Dependencies](#32-javascriptnextjs-dependencies-packagejson)
4. [Project Architecture](#4-project-architecture)
5. [File-by-File, Line-by-Line Explanation](#5-file-by-file-line-by-line-explanation)
   - [constants.py](#51-constantspy)
   - [embed_listings.py](#52-embed_listingspy)
   - [core.py](#53-corepy)
   - [server.py](#54-serverpy)
   - [\_\_init\_\_.py](#55-__init__py)
   - [requirements.txt](#56-requirementstxt)
   - [web/lib/types.ts](#57-weblibtypests)
   - [web/app/globals.css](#58-webappglobalscss)
   - [web/app/layout.tsx](#59-webapplayouttsx)
   - [web/app/page.tsx](#510-webapppagetsx)
   - [web/components/SearchPanel.tsx](#511-webcomponentssearchpaneltsx)
   - [web/next.config.ts](#512-webnextconfigts)
   - [web/postcss.config.mjs](#513-webpostcssconfigmjs)
   - [web/tsconfig.json](#514-webtsconfigjson)
   - [web/.env.local.example](#515-webenvlocalexample)
   - [web/package.json](#516-webpackagejson)
6. [How to Run the Project](#6-how-to-run-the-project)
7. [Key Concepts Glossary](#7-key-concepts-glossary)
8. [Next Steps for Learning](#8-next-steps-for-learning)

---

## 1. Project Overview

### What does this project do?

This project lets you **search Florida real estate listings by meaning**. Instead of typing an exact keyword like "pool" and only seeing listings that literally contain the word "pool," you can type something like "backyard oasis for entertaining" and the system will find listings whose *descriptions feel similar in meaning* — even if they never use the exact words you typed.

### What real-world problem does it solve?

Traditional search engines on real estate websites are **keyword-based**: they scan listing text for the exact words you type. If a seller wrote "resort-style outdoor living" but you searched "big backyard with pool," a keyword search misses it. **Semantic search** understands that those phrases are about similar things, because it compares the *meaning* of sentences rather than their exact letters.

### Analogy

Think of a **librarian** who has read every book in the library. You walk in and say, "I want something about a family moving to a small town and finding community." The librarian does not flip through indexes looking for those exact words — instead, she *understands the vibe* of what you want and hands you the right books. This project builds that kind of librarian, but for Florida real estate listings. The "understanding" comes from a machine-learning model that converts text into numbers (called **vectors** or **embeddings**), and the "finding the right books" comes from a database that is extremely fast at comparing those numbers.

---

## 2. Prerequisites & Environment

### What is a programming language?

A programming language is a set of rules that lets you write instructions a computer can follow. This project uses **two** programming languages:

- **Python** (version 3.10 or higher) — runs the backend: reads data, creates embeddings, stores vectors, serves the search API.
- **TypeScript** (a strict version of JavaScript) — runs the frontend website that you see in your browser.

### What is a terminal / command line?

The **terminal** (also called the command line, shell, or console) is a text-based window where you type commands instead of clicking buttons. You need it to:

- Install software and libraries.
- Run your Python scripts.
- Start the web server.
- Build and run Docker containers.

On macOS you can open it with **Terminal.app** or the integrated terminal in Cursor/VS Code (`` Ctrl+` `` or `` Cmd+` ``).

### Step-by-step setup

#### A. Install Python

1. Check if Python is already installed:

```bash
python3 --version
```

You should see something like `Python 3.12.x`. If not, download it from [python.org](https://www.python.org/downloads/) or install via Homebrew:

```bash
brew install python
```

#### B. Install Node.js (for the Next.js frontend)

```bash
node --version
```

You need Node.js 18 or higher. If you do not have it, install via [nodejs.org](https://nodejs.org/) or:

```bash
brew install node
```

#### C. Create a Python virtual environment

A **virtual environment** is an isolated box for your Python packages so they do not clash with other projects on your machine. Think of it as giving this project its own private toolbox.

From the **repo root** (`machine_learning/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- `python3 -m venv .venv` — tells Python to create a new virtual environment in a folder called `.venv`.
- `source .venv/bin/activate` — switches your terminal session into that environment. Your prompt will change (often showing `(.venv)` at the start) to remind you.

#### D. Install Python dependencies

```bash
pip install -r semantic_search/requirements.txt
```

This reads the list of libraries in `requirements.txt` and downloads them all.

#### E. Install Next.js dependencies

```bash
cd semantic_search/web
npm install
```

`npm install` reads `package.json` and downloads every JavaScript library listed there into a `node_modules/` folder.

---

## 3. Dependencies — Deep Dive

### 3.1 Python Dependencies (`requirements.txt`)

#### chromadb (>= 0.4.22)

- **What it is:** An open-source **vector database**. A vector database is specialized for storing and quickly searching through collections of numerical arrays (vectors).
- **What problem it solves:** If you had 10,000 listings, each represented as a list of 384 numbers, you *could* loop through all 10,000 and calculate distance yourself — but that would be painfully slow and would not scale. ChromaDB uses an efficient algorithm called **HNSW** (Hierarchical Navigable Small World) that finds the closest vectors without checking every single one.
- **Analogy:** ChromaDB is like a filing cabinet with a smart index — instead of opening every drawer to find the closest match, it knows which drawers to skip.
- **Used in:** `embed_listings.py` (to store vectors), `core.py` (to query vectors), `server.py` (indirectly, through `core.py`).

#### sentence-transformers (>= 2.2.0)

- **What it is:** A Python library that wraps Hugging Face transformer models and makes it dead easy to convert any sentence into a fixed-length vector of numbers (an **embedding**).
- **What problem it solves:** The raw transformer model outputs are complicated tensors. `sentence-transformers` handles tokenization, padding, pooling, and normalization so you can just call `model.encode("some text")` and get a neat list of 384 numbers back.
- **Analogy:** If the raw transformer model is a full orchestra, `sentence-transformers` is the conductor who organizes everything so you just hear a clean, finished piece of music.
- **Used in:** `embed_listings.py` (encoding all listing descriptions) and `core.py` (encoding the user's search query at query time).


#### pandas (>= 2.0.0)

- **What it is:** The standard Python library for working with tabular data (rows and columns), like spreadsheets or CSV files.
- **What problem it solves:** Python's built-in `csv` module can read CSV files, but it gives you raw strings. Pandas automatically detects data types, lets you filter rows, handle missing values, and perform calculations in one-liners.
- **Analogy:** Pandas is like a supercharged spreadsheet that lives inside your code — you can sort, filter, and transform columns without clicking through menus.
- **Used in:** `embed_listings.py` (loading the CSV of Florida listings, filtering out empty rows).

#### fastapi (>= 0.115.0)

- **What it is:** A modern Python web framework for building **APIs** (Application Programming Interfaces). An API is a set of URLs that other programs (not humans) call to get data back, usually in JSON format.
- **What problem it solves:** You need a way for the Next.js frontend (running in the browser) to ask the Python backend "give me search results for this query." FastAPI provides the URL endpoints, validates incoming data, and sends back structured JSON responses.
- **Analogy:** FastAPI is like a restaurant's order window — the kitchen (your Python search logic) does the cooking, and FastAPI is the window where orders come in and plates go out.
- **Used in:** `server.py` (defines the `/search` and `/health` endpoints).

#### uvicorn[standard] (>= 0.30.0)

- **What it is:** An **ASGI server** — the actual program that listens for network connections on a port (like port 8000) and passes incoming HTTP requests to your FastAPI code.
- **What problem it solves:** FastAPI defines *what* to do with requests, but it cannot listen to the network by itself. Uvicorn handles the low-level networking: opening a socket, accepting connections, speaking HTTP.
- **Analogy:** If FastAPI is the chef, Uvicorn is the waiter who stands at the door, takes orders from customers, and brings them to the kitchen.
- **Used in:** The command line (you run `uvicorn semantic_search.server:app ...`) and in production (your host's start command).

### 3.2 JavaScript/Next.js Dependencies (`package.json`)

#### next (16.2.2)

- **What it is:** A React-based framework for building websites. It handles routing (which URL shows which page), server-side rendering, bundling, and many optimizations automatically.
- **What problem it solves:** Plain React gives you components, but you still have to set up a build tool, a router, code splitting, and more. Next.js gives you all of that out of the box.
- **Analogy:** If React is a pile of LEGO bricks, Next.js is the instruction manual and the baseplate — it tells you where the pieces go so you get a finished house quickly.
- **Used in:** The entire `web/` folder. Every `.tsx` file runs inside the Next.js framework.

#### react (19.2.4) and react-dom (19.2.4)

- **What they are:** React is a JavaScript library for building user interfaces out of reusable "components." `react-dom` is the piece that takes those components and actually renders them into the browser's HTML.
- **What problem they solve:** Manually updating the HTML whenever data changes is tedious and error-prone. React keeps a virtual copy of the page, figures out what changed, and updates only those parts.
- **Analogy:** React is like a stage director who watches the script (your data) and tells each actor (HTML element) exactly when to move, without rearranging the whole stage.
- **Used in:** Every `.tsx` component file (`page.tsx`, `layout.tsx`, `SearchPanel.tsx`).

#### tailwindcss (^4) and @tailwindcss/postcss (^4)

- **What they are:** Tailwind CSS is a "utility-first" CSS framework. Instead of writing custom CSS classes like `.big-blue-button`, you compose tiny utility classes directly in your HTML: `className="bg-blue-500 text-white px-4 py-2"`.
- **What problem they solve:** Writing and maintaining a separate CSS file with hundreds of custom class names is messy. Tailwind gives you a consistent design system via small, composable classes.
- **Analogy:** Instead of buying a pre-made outfit (a CSS framework like Bootstrap), Tailwind gives you a huge wardrobe of individual clothing items (utility classes) and you pick exactly what to wear.
- **Used in:** Every `.tsx` file's `className` attributes and `globals.css`.

#### typescript (^5)

- **What it is:** A superset of JavaScript that adds **static types**. Types let you declare that a variable must be a string, a number, a specific shape of object, etc.
- **What problem it solves:** JavaScript lets you accidentally pass a number where a string was expected. TypeScript catches these mistakes *before* you run the code.
- **Analogy:** TypeScript is like spell-check for your code — it underlines mistakes before you hit send.
- **Used in:** Every `.ts` and `.tsx` file.

#### eslint (^9) and eslint-config-next

- **What they are:** ESLint is a code-quality tool that reads your JavaScript/TypeScript and warns about common mistakes, unused variables, and style inconsistencies. `eslint-config-next` is a set of rules tailored for Next.js projects.
- **Used in:** The `npm run lint` command, and your editor's real-time warning squiggles.

#### @types/node, @types/react, @types/react-dom

- **What they are:** Type definitions for Node.js, React, and ReactDOM. Since those libraries are written in JavaScript (not TypeScript), these `@types` packages tell TypeScript what shapes their functions and objects have.
- **Used in:** Behind the scenes — TypeScript reads these automatically so it can check your code.

---

## 4. Project Architecture

### 4.1 Folder and file structure

```
semantic_search/
├── __init__.py              ← Makes this folder a Python package
├── constants.py             ← Single source of truth: file paths, model name, collection name
├── embed_listings.py        ← Offline script: CSV → embeddings → ChromaDB
├── core.py                  ← Shared search logic (used by server.py)
├── server.py                ← FastAPI REST API (serves /search and /health)
├── requirements.txt         ← Python library versions
├── chroma_data/             ← Generated vector database files (created by embed_listings.py)
├── README.md                ← Original project overview
├── LEARN.md                 ← This file (study guide)
└── web/                     ← Next.js frontend (TypeScript + React + Tailwind)
    ├── package.json         ← JavaScript dependency list and scripts
    ├── tsconfig.json        ← TypeScript compiler settings
    ├── next.config.ts       ← Next.js configuration (currently empty/default)
    ├── postcss.config.mjs   ← PostCSS plugin config (enables Tailwind CSS v4)
    ├── .env.local.example   ← Template for environment variables
    ├── app/
    │   ├── layout.tsx       ← Root HTML shell (fonts, metadata, body wrapper)
    │   ├── page.tsx         ← Home page (renders SearchPanel)
    │   └── globals.css      ← Global styles (Tailwind import + CSS variables)
    ├── components/
    │   └── SearchPanel.tsx  ← The interactive search form and results list
    └── lib/
        └── types.ts         ← TypeScript type definitions matching the API response
```

### 4.2 Data flow (the order things happen)

Here is the journey of data through the system, step by step:

```
                     ONE-TIME INGESTION
                     ==================
 ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
 │  CSV file    │───▶│ embed_listings.py │───▶│  ChromaDB    │
 │ (10k rows)   │    │ reads CSV,       │    │ (chroma_data │
 │              │    │ creates vectors, │    │  on disk)    │
 └──────────────┘    │ stores them      │    └──────────────┘
                     └──────────────────┘

                     EVERY SEARCH QUERY
                     ==================
 ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
 │  User types  │───▶│ Next.js frontend │───▶│ FastAPI      │
 │  a query in  │    │ (SearchPanel.tsx) │    │ (server.py)  │
 │  the browser │    │ sends POST       │    │ calls core.py│
 └──────────────┘    │ /search request  │    │ which queries│
                     └──────────────────┘    │ ChromaDB     │
                                             └──────┬───────┘
                                                    │
                                             returns JSON
                                             (results list)
                                                    │
                                             ┌──────▼───────┐
                                             │ Browser shows │
                                             │ ranked results│
                                             └──────────────┘
```

1. **Ingestion (once):** You run `embed_listings.py`. It reads the CSV, sends every listing description through the `all-MiniLM-L6-v2` model to get a 384-number vector, and stores those vectors (plus the original text and metadata) in ChromaDB on disk.
2. **API startup:** You run `uvicorn`. It loads `server.py`, which creates a FastAPI app with two endpoints: `/health` and `/search`.
3. **User searches:** The user opens the Next.js website, types a query, and hits "Search." The browser sends a POST request to `/search` with the query text.
4. **Backend processing:** `server.py` receives the request, calls `core.py`'s `search_listings()`. That function encodes the query text into a vector using the *same* model and asks ChromaDB "give me the 5 closest vectors." ChromaDB returns IDs, texts, metadata, and distances.
5. **Response:** FastAPI sends the results back as JSON. The browser renders them as cards.

### 4.3 Import / dependency graph

```
constants.py          ← imported by everything below
    ▲
    │
    ├── embed_listings.py   (also imports: pandas, sentence_transformers, chromadb)
    │
    ├── core.py             (also imports: sentence_transformers, chromadb)
    │       ▲
    │       │
    │       ├── server.py       (also imports: fastapi, pydantic, os, sys)
    │       │
    │       └── (Next.js only; no Streamlit)
    │
    └── (Next.js frontend does NOT import Python files —
         it talks to server.py over HTTP)

web/lib/types.ts      ← imported by SearchPanel.tsx
web/components/SearchPanel.tsx ← imported by page.tsx
web/app/globals.css   ← imported by layout.tsx
```

---

## 5. File-by-File, Line-by-Line Explanation

---

### 5.1 `constants.py`

#### a. Purpose

This file exists so that the **embedding model name**, the **CSV file path**, the **ChromaDB storage folder**, and the **collection name** are defined in exactly one place. Both the ingestion script (`embed_listings.py`) and the search logic (`core.py`) import these values. If they used different model names or different collection names, the system would silently break — queries would be encoded in a different "language" than stored vectors, and results would be garbage.

#### b. Imports

```python
from pathlib import Path
```

- **`pathlib`** is a built-in Python module (no install needed) that provides `Path` objects — a modern, cross-platform way to work with file and folder paths. Instead of gluing strings together with `/` and hoping it works on Windows vs. Mac, `Path` does it correctly everywhere.
- **Why here?** We need to build file paths to the CSV, the ChromaDB folder, etc.

#### c. Line-by-line walkthrough

```python
"""
semantic_search/constants.py
------------------------------
Single source of truth for paths and model name.
...
"""
```

This is a **module docstring** — a multi-line string at the top of the file that documents what the file does. Python ignores it at runtime, but tools like `help()` and documentation generators read it. It explains *why* this file exists: to prevent the model or path from "drifting" between ingest and search.

---

```python
from pathlib import Path
```

Imports the `Path` class from the built-in `pathlib` module. We will use it to build file system paths.

---

```python
_SEMANTIC_DIR = Path(__file__).resolve().parent
```

Let's break this apart:

- **`__file__`** — a special built-in variable that Python sets automatically. It holds the path to the current file (in this case, something like `/Users/sedly/Desktop/machine_learning/semantic_search/constants.py`).
- **`Path(__file__)`** — wraps that string in a `Path` object so you can call methods on it.
- **`.resolve()`** — converts any relative path to an absolute path and resolves symlinks. This ensures the path is reliable no matter which working directory you ran the script from.
- **`.parent`** — gives you the *parent directory* (the folder containing this file). So if the file is `.../semantic_search/constants.py`, `.parent` is `.../semantic_search/`.
- **The leading underscore** in `_SEMANTIC_DIR` is a Python naming convention that means "this variable is private to this module — other files should not import it directly."

**Why?** We need a reliable anchor point. Everything else (CSV path, Chroma folder) is defined relative to this location.

---

```python
_REPO_ROOT = _SEMANTIC_DIR.parent
```

Goes up one more directory level. If `_SEMANTIC_DIR` is `.../machine_learning/semantic_search/`, then `_REPO_ROOT` is `.../machine_learning/`. This is where the CSV file lives.

---

```python
CSV_PATH = _REPO_ROOT / "Florida Real Estate Sold Properties.csv"
```

The `/` operator on `Path` objects joins path segments (like `os.path.join` but cleaner). This builds the full path to the CSV data file. Notice the variable name is ALL_CAPS — this is a Python convention for **constants** (values that should never be reassigned after they are set).

---

```python
CHROMA_DIR = _SEMANTIC_DIR / "chroma_data"
```

The folder where ChromaDB will store its database files (SQLite database, HNSW index, etc.). It lives inside `semantic_search/` so it travels with the project.

---

```python
COLLECTION_NAME = "florida_listings"
```

A single ChromaDB database can hold multiple **collections** (think of them as separate tables). This constant names our collection. Both the ingestion script (which creates it) and the search logic (which reads from it) must use the same name.

---

```python
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

The identifier of the embedding model on Hugging Face. `all-MiniLM-L6-v2` is a small, fast model that produces 384-dimensional vectors. "All" means it was trained on a diverse mix of data, "MiniLM" is the architecture (a smaller version of BERT), "L6" means 6 layers, and "v2" is the version.

**Why is this a constant?** If you embedded your listings with one model and then queried with a different model, the numbers would live in completely different mathematical spaces. It would be like measuring distances in kilometers on one side and miles on the other — comparisons would be meaningless.

#### d. How this file connects to the rest

- **Imported by:** `embed_listings.py`, `core.py`, `server.py`.
- **Imports from:** nothing in this project (only the built-in `pathlib`).

---

### 5.2 `embed_listings.py`

#### a. Purpose

This is the **ingestion pipeline** — the "offline" script you run once (or whenever your data changes) to convert raw CSV text into searchable vectors. It reads every listing description from the CSV, runs each one through the sentence-transformer model to get a numerical vector, and stores those vectors (along with the original text and metadata like price, sqft, beds, baths, ZIP code) in ChromaDB. After this script finishes, the `chroma_data/` folder exists on disk and the search engine can work.

#### b. Imports

```python
from __future__ import annotations
```

This is a **future import**. It tells Python to treat all type hints in this file as strings rather than evaluating them immediately. This lets you use newer type-hint syntax (like `str | int` instead of `Union[str, int]`) even on slightly older Python versions. It must be the very first import in the file.

---

```python
import argparse
```

**`argparse`** is a built-in Python module for parsing command-line arguments. When you run `python embed_listings.py --limit 100`, `argparse` is what reads `--limit 100` and gives you `args.limit = 100`. Without it, you would have to manually inspect `sys.argv` and handle errors yourself.

---

```python
import sys
from pathlib import Path
```

- **`sys`** — built-in module that gives access to Python internals, including `sys.path` (the list of directories Python searches when you write `import something`).
- **`Path`** — same as in `constants.py`, used for file path manipulation.

---

```python
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
```

This is a **path hack** to ensure Python can find `constants.py`. When you run `python semantic_search/embed_listings.py` from the repo root, Python's current working directory is the repo root, but `constants.py` lives inside `semantic_search/`. By inserting the script's own directory at the front of `sys.path`, the `from constants import ...` line below will succeed. Without this, Python would raise `ModuleNotFoundError: No module named 'constants'`.

---

```python
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
```

- **`chromadb`** — the vector database library.
- **`pandas as pd`** — the `as pd` part creates a shorter alias. Writing `pd.read_csv(...)` is quicker and is the universally recognized convention in the Python data world.
- **`SentenceTransformer`** — the class from the `sentence-transformers` library that loads a model and provides the `.encode()` method.

---

```python
from constants import CHROMA_DIR, COLLECTION_NAME, CSV_PATH, DEFAULT_MODEL
```

Imports the four constants we defined in `constants.py`. This is the payoff of having a single source of truth — if you change the model name, you only change it in one place.

---

```python
BATCH_SIZE = 64
```

When encoding thousands of texts, sending them to the model one at a time is slow. Sending them all at once might exceed your RAM. `BATCH_SIZE = 64` means "send 64 texts to the model at a time." This is a **memory vs. speed tradeoff**: bigger batches are faster (the GPU/CPU processes them in parallel) but use more memory.

#### c. Line-by-line walkthrough

##### The `_scalar_meta` helper function

```python
def _scalar_meta(val) -> str | int | float | bool:
```

This defines a function called `_scalar_meta` (the leading underscore means "private — only used inside this file"). The `-> str | int | float | bool` part is a **type hint** (also called a return type annotation) that says "this function returns either a string, an integer, a float, or a boolean." Type hints do not change how the code runs; they are documentation for humans and for tools like mypy.

---

```python
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
```

**Why this exists:** CSV data is messy. Some cells are empty, which pandas represents as `NaN` (Not a Number, a special float value). ChromaDB's metadata system **rejects** `NaN` and numpy-specific types. So this function checks: if the value is `None` or is a float that is `NaN`, return an empty string instead.

- **`isinstance(val, float)`** checks if `val` is a float.
- **`pd.isna(val)`** is a pandas function that returns `True` if the value is `NaN`.
- **`is None`** checks for Python's `None` value (the "nothing" value).

---

```python
    if isinstance(val, bool):
        return val
```

Booleans (`True` / `False`) are fine as-is. This check must come *before* the `int` check because in Python, `bool` is a subclass of `int` — `isinstance(True, int)` returns `True`, which would cause booleans to be converted to `1` or `0` if we checked `int` first.

---

```python
    if isinstance(val, (int, float)):
        return float(val) if isinstance(val, float) else int(val)
```

If the value is a number, keep it as a number — but make sure it is a plain Python `int` or `float`, not a numpy `int64` or `float32` (which ChromaDB does not accept). The **ternary expression** `a if condition else b` is Python's inline if/else: "return `float(val)` if it is a float, otherwise return `int(val)`."

---

```python
    return str(val)
```

Anything else (strings, objects, unexpected types) — convert to a string. This is the safety net.

---

##### The `main()` function

```python
def main() -> None:
```

Defines the main function. `-> None` means it does not return a value; it performs actions (side effects) and then finishes.

---

```python
    parser = argparse.ArgumentParser(description="Embed MLS texts into ChromaDB.")
```

Creates an **argument parser** — an object that will define what command-line flags the script accepts and then parse them from `sys.argv`.

---

```python
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Must stay in sync with core.py / constants.py",
    )
```

Defines an optional flag `--model`. If the user does not provide it, it defaults to `DEFAULT_MODEL` (from `constants.py`). The `help` text shows up if the user runs `python embed_listings.py --help`.

---

```python
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N rows (omit for full ~10k dataset).",
    )
```

Another optional flag. `type=int` means argparse will convert the user's string input ("100") to an integer (100). `default=None` means "embed all rows" if the flag is omitted.

---

```python
    args = parser.parse_args()
```

Actually reads the command-line arguments and stores them in `args`. After this, `args.model` and `args.limit` are available.

---

```python
    if not CSV_PATH.is_file():
        raise SystemExit(f"CSV not found: {CSV_PATH}")
```

**Fail-fast pattern:** Before doing anything expensive (loading the model, etc.), check that the data file exists. `is_file()` returns `True` only if the path points to an existing file (not a folder, not missing). `raise SystemExit(...)` immediately stops the script and prints the error message. The `f"..."` syntax is an **f-string** (formatted string) — Python replaces `{CSV_PATH}` with the actual path value.

---

```python
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
```

- `pd.read_csv(...)` reads the CSV file into a **DataFrame** — pandas' primary data structure, like a table with named columns and numbered rows.
- `encoding="utf-8-sig"` handles a **BOM** (Byte Order Mark) — an invisible character some programs (like Excel) put at the start of CSV files. Without this, the first column name might have garbage characters prepended.
- `low_memory=False` tells pandas to read the entire file at once to determine column types, rather than guessing from chunks (which can cause mixed-type warnings).

---

```python
    if "sanitized_text" not in df.columns:
        raise SystemExit("Expected column 'sanitized_text' in CSV.")
```

Another fail-fast check. The column `sanitized_text` is where the listing descriptions live. If the CSV does not have it, stop immediately with a clear error rather than crashing later with a confusing `KeyError`.

---

```python
    df = df[df["sanitized_text"].notna() & (df["sanitized_text"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)
```

- `df["sanitized_text"].notna()` creates a boolean column: `True` for rows where the text is not `NaN`.
- `.astype(str).str.strip() != ""` converts to string, strips whitespace, and checks it is not empty.
- `&` combines both conditions (AND). Only rows passing *both* checks survive.
- `df[...]` with a boolean array filters to matching rows (this is called **boolean indexing**).
- `.reset_index(drop=True)` re-numbers the rows from 0 to N-1. `drop=True` means "throw away the old index instead of adding it as a column."

**Why?** Empty descriptions cannot be meaningfully embedded. The model would produce a generic, near-zero vector that pollutes search results.

---

```python
    if args.limit is not None:
        df = df.head(args.limit)
```

If the user passed `--limit 100`, keep only the first 100 rows. `.head(n)` returns the first `n` rows. This is useful for quick testing — embedding all 10,000 rows can take several minutes.

---

```python
    n = len(df)
    print(f"Rows to embed: {n}")
```

`len(df)` returns the number of rows. Printing it tells the user how much work is about to happen.

---

```python
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
```

This line downloads (on first run) and loads the embedding model into memory. The model weights are hundreds of megabytes. After the first download, they are cached locally so subsequent runs are fast.

---

```python
    texts = df["sanitized_text"].astype(str).tolist()
```

Extracts the text column as a plain Python list of strings. `.astype(str)` ensures everything is a string (not `NaN`). `.tolist()` converts from a pandas Series to a regular list, which `sentence-transformers` expects.

---

```python
    print("Encoding (this may take a few minutes on full data)…")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
```

This is the most important line in the file. Let's break down each parameter:

- **`texts`** — the list of listing descriptions to encode.
- **`batch_size=BATCH_SIZE`** — process 64 texts at a time (the tradeoff discussed earlier).
- **`show_progress_bar=True`** — displays a progress bar in the terminal so you know how far along it is.
- **`convert_to_numpy=True`** — returns a numpy array (a fast, memory-efficient matrix) rather than a list of lists.
- **`normalize_embeddings=True`** — scales every vector to have length 1 (a **unit vector**). This is critical because when vectors are normalized, **cosine distance** equals a simple formula and "closer" always means "more similar." Without normalization, a longer vector could appear closer just because it has bigger numbers, not because it is more similar in meaning.

The result `embeddings` is a matrix of shape `(n, 384)` — `n` rows (one per listing) and 384 columns (one per dimension of the vector).

---

```python
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
```

- `.mkdir(parents=True, exist_ok=True)` creates the `chroma_data/` folder and any parent directories if they do not exist. `exist_ok=True` means "don't error if it already exists."
- `PersistentClient` opens (or creates) a ChromaDB database that is persisted to disk. Without "Persistent," it would only live in memory and disappear when the script ends.

---

```python
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
```

If you are re-running the script (maybe you changed the data or the model), you want a clean slate. This deletes the old collection. The `try/except` block catches the error that occurs if the collection does not exist yet (first run). `pass` means "do nothing" in the except block.

---

```python
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
```

Creates a new, empty collection. The `metadata={"hnsw:space": "cosine"}` tells ChromaDB's HNSW index to measure distance using **cosine distance** (which measures the angle between two vectors). Since our vectors are normalized, cosine distance ranges from 0 (identical) to 2 (opposite). Other options would be "l2" (Euclidean distance) or "ip" (inner product), but cosine is the standard for text embeddings.

---

```python
    ids = [f"row_{i}" for i in range(n)]
```

Creates a list of unique ID strings: `["row_0", "row_1", "row_2", ...]`. ChromaDB requires every stored item to have a unique ID. This is a **list comprehension** — a compact way to build a list by looping.

---

```python
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append(
            {
                "lastSoldPrice": _scalar_meta(row.get("lastSoldPrice")),
                "listPrice": _scalar_meta(row.get("listPrice")),
                "sqft": _scalar_meta(row.get("sqft")),
                "beds": _scalar_meta(row.get("beds")),
                "baths": _scalar_meta(row.get("baths")),
                "zip": _scalar_meta(row.get("zip")),
                "type": _scalar_meta(row.get("type")),
            }
        )
```

Builds a list of dictionaries, one per row. Each dictionary holds metadata about the listing (price, sqft, beds, baths, ZIP code, type). Every value is passed through `_scalar_meta()` to ensure ChromaDB will accept it.

- **`df.iterrows()`** loops through the DataFrame row by row. It yields a tuple of `(index, row)`. The `_` is a Python convention meaning "I don't need this value" (we ignore the index).
- **`row.get("beds")`** safely retrieves the "beds" column value. If the column does not exist, it returns `None` instead of crashing.

---

```python
    chunk = 500
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Indexed {end}/{n}")
```

Adds data to ChromaDB in **chunks** of 500 items. Why not all at once? Sending 10,000 items in a single network/database call can be slow or cause memory spikes. Chunking keeps things manageable.

- **`range(0, n, chunk)`** generates `0, 500, 1000, 1500, ...` — the start of each chunk.
- **`min(start + chunk, n)`** ensures the last chunk does not go past the end of the data.
- **Slice notation `[start:end]`** selects items from index `start` up to (but not including) `end`.
- **`.tolist()`** converts the numpy array slice to a plain Python list (ChromaDB expects lists).
- **`collection.add(...)`** inserts items into the collection. Each item has an ID, an embedding vector, the original document text, and metadata.

---

```python
    print(f"Done. Vector store: {CHROMA_DIR}")
    print("Next: uvicorn semantic_search.server:app --host 0.0.0.0 --port 8000")
```

Success messages. The second line tells the user what to do next.

---

```python
if __name__ == "__main__":
    main()
```

**The `if __name__ == "__main__"` guard:** When you run a Python file directly (`python embed_listings.py`), Python sets a special variable `__name__` to the string `"__main__"`. When the file is *imported* by another file, `__name__` is set to the module name instead. This guard ensures `main()` only runs when the file is executed directly, not when it is imported. It is the standard entry point pattern in Python.

#### d. How this file connects to the rest

- **Imports from:** `constants.py` (paths, model name), plus `chromadb`, `pandas`, `sentence_transformers`.
- **Called by:** You, manually, from the terminal. No other file in this project runs it.
- **Produces:** The `chroma_data/` folder, which `core.py` (and therefore `server.py`) depend on.

---

### 5.3 `core.py`

#### a. Purpose

This file contains the **shared search logic** — the function `search_listings()` that `server.py` (the API) calls. By putting the search logic in one place, you avoid duplicating code and ensure both frontends produce identical results.

#### b. Imports

```python
from __future__ import annotations
```

Same future import as before — enables modern type hint syntax.

---

```python
from functools import lru_cache
```

**`functools`** is a built-in module with tools for working with functions. **`lru_cache`** is a **decorator** that caches (remembers) the return value of a function based on its arguments. "LRU" stands for **Least Recently Used** — when the cache is full, the entry that has not been used for the longest time gets evicted.

**Why here?** Loading the embedding model takes several seconds and uses lots of memory. If every search query re-loaded the model, the app would be painfully slow. `lru_cache` ensures the model is loaded once and then reused.

---

```python
import chromadb
from sentence_transformers import SentenceTransformer
```

Same as in `embed_listings.py` — we need chromadb to query the vector store, and `SentenceTransformer` to encode the user's query.

---

```python
from constants import CHROMA_DIR, COLLECTION_NAME, DEFAULT_MODEL
```

Imports the same constants, ensuring we use the same collection and model as ingestion.

#### c. Line-by-line walkthrough

```python
MAX_K = 20
```

The maximum number of results any single search can return. This prevents a user from requesting thousands of results (which would be slow and pointless).

---

```python
_collection = None
```

A **module-level variable** that will hold the ChromaDB collection object after it is first opened. Starting as `None` means "not loaded yet." The underscore prefix signals it is private.

---

```python
def vector_store_ready() -> bool:
    """True when chroma_data/ exists (ingest has been run at least once)."""
    return CHROMA_DIR.is_dir()
```

A simple check: does the `chroma_data/` directory exist? If not, the user needs to run `embed_listings.py` first. `.is_dir()` returns `True` if the path is an existing directory.

---

```python
def get_collection():
    """Lazy-open the Chroma persistent client and return the collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection
```

This is the **lazy initialization** pattern (also called a **singleton**):

- **`global _collection`** — tells Python that when we assign to `_collection` inside this function, we mean the module-level variable, not a new local variable. Without `global`, Python would create a local variable and the module-level one would stay `None` forever.
- The `if _collection is None` check means we only open the database on the *first* call. Every subsequent call returns the already-opened collection instantly.
- **`client.get_collection(...)`** retrieves an existing collection by name (the one created by `embed_listings.py`).

---

```python
@lru_cache(maxsize=4)
def _get_model(model_id: str) -> SentenceTransformer:
    """Load (and cache) the embedding model so repeat queries skip download."""
    return SentenceTransformer(model_id)
```

The **`@lru_cache(maxsize=4)`** is a **decorator** — a special annotation that wraps the function with extra behavior. Here, it means:

- The first time you call `_get_model("sentence-transformers/all-MiniLM-L6-v2")`, it actually downloads/loads the model and returns it. The result is then stored in a cache.
- The second time you call it with the same argument, it returns the cached model instantly — no loading.
- `maxsize=4` means it can cache up to 4 different models. If you loaded a 5th, the least-recently-used one would be evicted.

---

```python
def search_listings(
    query: str,
    k: int = 5,
    model_id: str | None = None,
) -> list[dict]:
```

The main search function. Let's look at the parameters:

- **`query: str`** — the text the user typed (a type hint saying it must be a string).
- **`k: int = 5`** — how many results to return. The `= 5` makes it optional with a default.
- **`model_id: str | None = None`** — optional override for the embedding model. `str | None` means it can be a string or `None`.
- **`-> list[dict]`** — the function returns a list of dictionaries.

---

```python
    q = query.strip()
    if not q:
        return []
```

`.strip()` removes leading and trailing whitespace. If the query is empty after stripping, return an empty list immediately — there is nothing to search for.

---

```python
    mid = model_id or DEFAULT_MODEL
    k = max(1, min(int(k), MAX_K))
```

- **`model_id or DEFAULT_MODEL`** — if `model_id` is `None` or empty, use `DEFAULT_MODEL`. The `or` operator in Python returns the first truthy value.
- **`max(1, min(int(k), MAX_K))`** — **clamps** `k` to the range [1, 20]. `min(k, 20)` ensures it does not exceed 20. `max(1, ...)` ensures it is at least 1. This is defensive programming — even if a caller passes `k=0` or `k=1000`, the value stays within safe bounds.

---

```python
    model = _get_model(mid)
    q_emb = model.encode(
        [q],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
```

Loads the cached model, then encodes the single query string into a vector. Note:

- **`[q]`** wraps the string in a list because `encode()` expects a list of strings (even for one string).
- **`normalize_embeddings=True`** is crucial — it must match what was used during ingestion. If ingestion normalized but search did not (or vice versa), cosine distances would be wrong.

The result `q_emb` is a numpy array of shape `(1, 384)` — one query, 384 dimensions.

---

```python
    raw = get_collection().query(
        query_embeddings=q_emb.tolist(),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
```

Asks ChromaDB: "Find the `k` vectors closest to my query vector." The `include` parameter specifies what extra data to return alongside the IDs:

- **`"documents"`** — the original listing text.
- **`"metadatas"`** — the price, sqft, etc. dictionaries.
- **`"distances"`** — how far each result is from the query (lower = more similar).

---

```python
    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]
```

ChromaDB returns results wrapped in an extra list (because you *could* query multiple queries at once). The `[0]` extracts the results for our single query.

---

```python
    out: list[dict] = []
    for doc_id, text, meta, dist in zip(ids, docs, metas, dists, strict=True):
```

- **`out: list[dict] = []`** — initializes an empty list with a type annotation.
- **`zip(..., strict=True)`** iterates over four lists in parallel, yielding one item from each list per iteration. `strict=True` means "raise an error if the lists have different lengths" — a safety check.

---

```python
        d = float(dist) if dist is not None else None
        sim = (1.0 - d) if d is not None else None
```

- Converts the distance to a Python `float` (it might be a numpy float).
- Computes **similarity** as `1 - distance`. For cosine distance with normalized vectors, distance ranges from 0 (identical) to 2 (opposite), so similarity ranges from -1 to 1. In practice, for text, similarity is usually between 0 and 1.

---

```python
        safe_meta = {str(k): v for k, v in (meta or {}).items() if v is not None}
```

A **dictionary comprehension** that filters out `None` values from metadata and ensures all keys are strings. `meta or {}` means "if meta is None, use an empty dict instead."

Note: the `k` inside this comprehension is a *different* variable from the function parameter `k` — it is the dictionary key in each key-value pair. This is a case where variable shadowing happens inside a comprehension scope.

---

```python
        out.append(
            {
                "id": doc_id,
                "text": text or "",
                "metadata": safe_meta,
                "distance": d,
                "similarity": sim,
            }
        )
    return out
```

Builds a dictionary for each result and appends it to the output list. Each dictionary has a consistent shape that `server.py` can convert directly into a JSON response.

#### d. How this file connects to the rest

- **Imports from:** `constants.py`, `chromadb`, `sentence_transformers`.
- **Imported by:** `server.py`.
- **Depends on:** The `chroma_data/` folder created by `embed_listings.py`.

---

### 5.4 `server.py`

#### a. Purpose

This file creates a **REST API** — a web service that other programs (like the Next.js frontend) can call over HTTP. It exposes two endpoints:

- `GET /health` — a quick check to see if the server is running and the vector store is ready.
- `POST /search` — accepts a JSON body with a search query and returns ranked results.

#### b. Imports

```python
from __future__ import annotations
```

Enables modern type hint syntax (same as other files).

---

```python
import os
import sys
from pathlib import Path
from typing import Any
```

- **`os`** — built-in module for operating system interactions. Used here to read **environment variables** (settings that are passed to the program from outside, like `ALLOW_ORIGINS`).
- **`sys`** — used for the path hack (same as `embed_listings.py`).
- **`Path`** — file path handling.
- **`typing.Any`** — a special type hint that means "any type at all." Used in the metadata field.

---

```python
_sem = Path(__file__).resolve().parent
if str(_sem) not in sys.path:
    sys.path.insert(0, str(_sem))
```

Same path hack as `embed_listings.py`, ensuring `constants` and `core` can be imported.

---

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
```

- **`FastAPI`** — the main class. You create an instance of it (`app = FastAPI(...)`) and then add routes to it.
- **`HTTPException`** — a special exception you can raise to return an HTTP error response (like 503 Service Unavailable).
- **`CORSMiddleware`** — middleware that handles **CORS** (Cross-Origin Resource Sharing). When your Next.js app at `localhost:3000` tries to call your API at `localhost:8000`, the browser blocks it by default for security. CORS headers tell the browser "it's okay, I trust this origin."
- **`BaseModel`** — from **Pydantic**, a library (bundled with FastAPI) for data validation. You define a class that inherits from `BaseModel`, and Pydantic automatically validates incoming JSON, converts types, and generates error messages.
- **`Field`** — lets you add constraints to `BaseModel` fields (like minimum length, default values, descriptions).

---

```python
from constants import CHROMA_DIR, COLLECTION_NAME, DEFAULT_MODEL
from core import MAX_K, get_collection, search_listings, vector_store_ready
```

Imports constants and the shared search logic.

#### c. Line-by-line walkthrough

```python
_DEFAULT_ORIGINS = "http://localhost:3000"
```

The default CORS origin — the address where the Next.js dev server runs. If no environment variable overrides it, only `localhost:3000` is allowed to call this API from a browser.

---

```python
def _cors_origins() -> list[str]:
    raw = os.environ.get("ALLOW_ORIGINS", _DEFAULT_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]
```

- **`os.environ.get("ALLOW_ORIGINS", _DEFAULT_ORIGINS)`** reads the environment variable `ALLOW_ORIGINS`. If it is not set, use the default.
- **`.split(",")`** splits the string by commas (in case you set multiple origins like `http://localhost:3000,https://myapp.vercel.app`).
- The list comprehension strips whitespace from each origin and filters out empty strings.

---

```python
app = FastAPI(title="Florida MLS semantic search", version="1.0.0")
```

Creates the FastAPI application object. The `title` and `version` show up in the auto-generated API documentation (available at `/docs` when the server is running).

---

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Adds the CORS middleware to the app. **Middleware** is code that runs on *every* request, before your endpoint function. This middleware adds the necessary HTTP headers so browsers allow cross-origin requests.

- **`allow_origins`** — which origins (websites) are allowed to call this API.
- **`allow_credentials=True`** — allows cookies/auth headers (not used here, but good practice).
- **`allow_methods=["*"]`** — allows all HTTP methods (GET, POST, PUT, DELETE, etc.).
- **`allow_headers=["*"]`** — allows all custom headers.

---

```python
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search")
    k: int = Field(default=5, ge=1, le=MAX_K)
    model_id: str | None = Field(default=None, description="Override embedding model")
```

A **Pydantic model** that defines the expected shape of the JSON body for POST /search. FastAPI uses this to:

1. **Validate** incoming requests (if `query` is missing or empty, FastAPI automatically returns a 422 error).
2. **Convert** types (if the client sends `k` as a string, Pydantic tries to convert it to an int).
3. **Document** the API (these descriptions appear in the auto-generated docs).

- **`Field(..., min_length=1)`** — the `...` (Ellipsis) means "this field is required, there is no default." `min_length=1` rejects empty strings.
- **`ge=1, le=MAX_K`** — "greater than or equal to 1, less than or equal to 20."

---

```python
class SearchHit(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None
    similarity: float | None
```

Defines the shape of a single search result. This is the **response model** — FastAPI uses it to serialize (convert to JSON) and validate outgoing data.

---

```python
class SearchResponse(BaseModel):
    results: list[SearchHit]
```

The full response: a list of hits wrapped in an object. Wrapping in an object (rather than returning a bare list) is a best practice because it is easier to extend later (you could add `total_count`, `page`, etc.).

---

```python
@app.get("/health")
def health():
```

The **`@app.get("/health")`** decorator tells FastAPI: "When someone sends a GET request to `/health`, run this function." This is called a **route** or **endpoint**.

---

```python
    ready = vector_store_ready()
    if ready:
        try:
            get_collection()
            return {"status": "ok", "ready": True, "chroma_dir": str(CHROMA_DIR)}
        except Exception as e:
            return {
                "status": "degraded",
                "ready": False,
                "error": str(e),
                "chroma_dir": str(CHROMA_DIR),
            }
    return {
        "status": "degraded",
        "ready": False,
        "detail": "Run embed_listings.py to create chroma_data",
        "chroma_dir": str(CHROMA_DIR),
    }
```

The health endpoint checks two things:

1. Does `chroma_data/` exist?
2. Can we actually open the collection?

If both succeed, it returns `"status": "ok"`. If either fails, it returns `"degraded"` with an explanation. This is useful for monitoring and debugging — if the frontend gets errors, you can hit `/health` to see what is wrong.

---

```python
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
```

- **`@app.post("/search")`** — POST requests to `/search` run this function.
- **`response_model=SearchResponse`** — tells FastAPI to validate and serialize the return value using `SearchResponse`. This also generates documentation.
- **`req: SearchRequest`** — FastAPI sees that the parameter type is a Pydantic model and automatically parses the JSON request body into it.

---

```python
    if not vector_store_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Vector store missing. Run embed_listings.py; expected {CHROMA_DIR}",
        )
```

If the vector store does not exist, return HTTP 503 (Service Unavailable). **`HTTPException`** is FastAPI's way of returning error responses. The `detail` string becomes the error message in the JSON response.

---

```python
    try:
        get_collection()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot open Chroma collection `{COLLECTION_NAME}`: {e}",
        ) from e
```

Tries to open the collection. If it fails (corrupted data, wrong collection name, etc.), return 503. The **`from e`** part chains the original exception for debugging.

---

```python
    mid = req.model_id or DEFAULT_MODEL
    hits = search_listings(req.query, k=req.k, model_id=mid)
    return SearchResponse(results=[SearchHit(**h) for h in hits])
```

- Calls the shared `search_listings()` function from `core.py`.
- **`SearchHit(**h)`** — the `**` operator unpacks the dictionary `h` as keyword arguments. If `h = {"id": "row_0", "text": "...", ...}`, then `SearchHit(**h)` becomes `SearchHit(id="row_0", text="...", ...)`.
- Returns a `SearchResponse` that FastAPI converts to JSON.

#### d. How this file connects to the rest

- **Imports from:** `constants.py`, `core.py`, `fastapi`, `pydantic`.
- **Called by:** Uvicorn (the ASGI server) loads this file.
- **Consumed by:** The Next.js frontend (`SearchPanel.tsx`), or any HTTP client (curl, Postman, etc.).

---

### 5.5 `__init__.py`

#### a. Purpose

```python
# Semantic search package (ingest, FastAPI, Next.js client).
```

This file is almost empty — it contains a single comment. Its *existence* is what matters: it tells Python that the `semantic_search/` folder is a **package** (a collection of modules that can be imported as a group). Without it, Python would not recognize `from semantic_search.server import app` (which is how uvicorn loads the API).

#### b. How this file connects to the rest

- **Imported by:** Uvicorn uses the package path `semantic_search.server:app`, which requires `__init__.py` to exist.
- **Does not import** anything itself.

---

### 5.6 `requirements.txt`

#### a. Purpose

```
# Project 1 — semantic search stack (local embeddings, no OpenAI key required)
chromadb>=0.4.22
sentence-transformers>=2.2.0
pandas>=2.0.0
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
```

This file lists every Python library the project needs, along with minimum version numbers.

- **`>=0.4.22`** means "version 0.4.22 or any newer version."
- **`uvicorn[standard]`** — the `[standard]` part is an **extras** specifier. It tells pip to install additional optional dependencies that uvicorn benefits from (like `httptools` for faster HTTP parsing and `uvloop` for faster async).

When you run `pip install -r requirements.txt`, pip reads this file and installs everything listed.

#### b. How this file connects to the rest

- **Used by:** `pip install` (and CI or container builds that install dependencies).
- **Defines:** All the libraries that the Python files import.

---

### 5.7 `web/lib/types.ts`

#### a. Purpose

This file defines **TypeScript types** that match the JSON shape the FastAPI backend sends. By defining them here, every component that uses search results gets autocompletion and type checking.

#### b. Line-by-line walkthrough

```typescript
/** Matches FastAPI SearchHit / SearchResponse */
```

A **JSDoc comment** — the `/** */` format. It documents what the types correspond to.

---

```typescript
export type SearchHit = {
  id: string;
  text: string;
  metadata: Record<string, unknown>;
  distance: number | null;
  similarity: number | null;
};
```

- **`export`** makes this type available for import by other files.
- **`type SearchHit = { ... }`** defines a **type alias** — a custom type with a specific shape. Any object assigned to a variable of type `SearchHit` must have exactly these fields.
- **`Record<string, unknown>`** is a TypeScript utility type meaning "an object whose keys are strings and whose values can be anything." This matches the flexible metadata dictionary from Python.
- **`number | null`** — a **union type** meaning "either a number or null." This matches Python's `float | None`.

---

```typescript
export type SearchResponse = {
  results: SearchHit[];
};
```

The overall API response shape: an object with a `results` field containing an array of `SearchHit` objects. `SearchHit[]` is TypeScript syntax for "an array of SearchHit."

#### c. How this file connects to the rest

- **Imported by:** `SearchPanel.tsx`.
- **Mirrors:** The Pydantic models `SearchHit` and `SearchResponse` in `server.py`.

---

### 5.8 `web/app/globals.css`

#### a. Purpose

This file defines the **global styles** that apply to every page of the Next.js app. It imports Tailwind CSS and sets up CSS custom properties (variables) for light and dark mode.

#### b. Line-by-line walkthrough

```css
@import "tailwindcss";
```

This single line pulls in the entire Tailwind CSS v4 framework. Tailwind v4 uses this `@import` approach (processed by PostCSS) instead of the older `@tailwind base; @tailwind components; @tailwind utilities;` directives. After this import, you can use any Tailwind utility class (like `bg-zinc-50`, `text-sm`, `px-4`) in your React components.

---

```css
:root {
  --background: #ffffff;
  --foreground: #171717;
}
```

**`:root`** selects the top-level HTML element. **CSS custom properties** (also called CSS variables) are defined with the `--` prefix. Here, `--background` is white and `--foreground` is a near-black color. These are the **light mode** colors.

---

```css
@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --font-sans: var(--font-geist-sans);
  --font-mono: var(--font-geist-mono);
}
```

**`@theme inline`** is a Tailwind CSS v4 directive that registers custom design tokens. It tells Tailwind: "When I use `bg-background` or `text-foreground` in my class names, use these CSS variable values." The `var(--font-geist-sans)` references a CSS variable that is set by the font loader in `layout.tsx`.

---

```css
@media (prefers-color-scheme: dark) {
  :root {
    --background: #0a0a0a;
    --foreground: #ededed;
  }
}
```

A **media query** that detects if the user's operating system is set to dark mode. If so, it overrides `--background` to near-black and `--foreground` to near-white. This gives the app automatic dark mode support.

---

```css
body {
  background: var(--background);
  color: var(--foreground);
  font-family: Arial, Helvetica, sans-serif;
}
```

Applies the background and text colors to the `<body>` element, and sets a fallback font stack.

#### c. How this file connects to the rest

- **Imported by:** `layout.tsx` (via `import "./globals.css"`).
- **Affects:** Every page and component in the app (it is global CSS).

---

### 5.9 `web/app/layout.tsx`

#### a. Purpose

This is the **root layout** — the outermost HTML shell that wraps every page. In Next.js's App Router, `layout.tsx` defines shared structure that persists across page navigations. It sets up the HTML document, loads fonts, applies metadata (browser tab title), and wraps the page content in a `<body>` tag.

#### b. Imports

```typescript
import type { Metadata } from "next";
```

Imports the `Metadata` type from Next.js. `import type` means "only import the type, not any runtime code" — this keeps the bundle smaller.

---

```typescript
import { Geist, Geist_Mono } from "next/font/google";
```

Next.js has a built-in font optimization system. `next/font/google` lets you load Google Fonts at build time — the font files are downloaded during the build and served locally, so there is no flash of unstyled text and no request to Google's servers at runtime.

- **`Geist`** — a clean, modern sans-serif font.
- **`Geist_Mono`** — its monospaced companion (for code snippets).

---

```typescript
import "./globals.css";
```

Imports the global stylesheet we just discussed. This is how CSS files are loaded in Next.js — you import them like JavaScript modules.

#### c. Line-by-line walkthrough

```typescript
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});
```

Initializes the two fonts. The `variable` option creates a CSS custom property (like `--font-geist-sans`) that holds the font family name. `subsets: ["latin"]` means "only include Latin characters" (not Cyrillic, Arabic, etc.) to keep the font file small.

---

```typescript
export const metadata: Metadata = {
  title: "Florida MLS semantic search",
  description: "Semantic search over listing descriptions via FastAPI + Chroma",
};
```

Next.js reads this `metadata` export and generates `<title>` and `<meta>` tags in the HTML `<head>`. The `title` appears in the browser tab; the `description` is used by search engines.

---

```typescript
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
```

This is a **React component** defined as a function. Let's unpack the syntax:

- **`export default`** makes this the file's main export, which Next.js uses as the layout.
- **`{ children }`** — **destructuring** the props object. `children` is a special React prop that contains whatever is nested inside this component (in this case, the page content).
- **`Readonly<{ children: React.ReactNode }>`** is a TypeScript type saying "props is an object with a `children` field, and the whole thing is read-only."
- **`React.ReactNode`** is the type for "anything React can render" — a string, a number, a component, an array of components, null, etc.

---

```typescript
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
```

Returns the HTML structure. This is **JSX** — a syntax extension that lets you write HTML-like code inside JavaScript/TypeScript.

- **`lang="en"`** tells the browser and screen readers the page is in English.
- **`className={...}`** — in JSX, you use `className` instead of HTML's `class` (because `class` is a reserved word in JavaScript). The template literal (backtick string) combines the font variables with Tailwind classes `h-full` (full height) and `antialiased` (smooth font rendering).
- **`min-h-full flex flex-col`** — Tailwind classes: minimum height 100%, flexbox layout, vertical direction.
- **`{children}`** renders whatever page component is currently active.

#### d. How this file connects to the rest

- **Imports:** `globals.css`, `next/font/google`, Next.js's `Metadata` type.
- **Wraps:** Every page in the app (including `page.tsx`).

---

### 5.10 `web/app/page.tsx`

#### a. Purpose

This is the **home page** — the component that renders when you visit the root URL (`/`). It is deliberately minimal: it just renders the `SearchPanel` component inside a `<main>` tag.

#### b. Line-by-line walkthrough

```typescript
import SearchPanel from "@/components/SearchPanel";
```

Imports the `SearchPanel` component. **`@/`** is a path alias defined in `tsconfig.json` — it means "the root of the `web/` folder." So `@/components/SearchPanel` resolves to `web/components/SearchPanel.tsx`.

---

```typescript
export default function Home() {
  return (
    <main className="min-h-full flex-1 bg-zinc-50 dark:bg-zinc-950">
      <SearchPanel />
    </main>
  );
}
```

- **`<main>`** is a semantic HTML element for the page's primary content.
- **`className="min-h-full flex-1 bg-zinc-50 dark:bg-zinc-950"`** — Tailwind classes:
  - `min-h-full` — minimum height 100%.
  - `flex-1` — grow to fill available space (works with the flexbox layout in `layout.tsx`).
  - `bg-zinc-50` — very light gray background.
  - `dark:bg-zinc-950` — very dark background in dark mode. The `dark:` prefix is a Tailwind modifier that applies only in dark mode.
- **`<SearchPanel />`** renders the search form and results list.

#### c. How this file connects to the rest

- **Imports:** `SearchPanel.tsx`.
- **Rendered inside:** `layout.tsx` (as `{children}`).

---

### 5.11 `web/components/SearchPanel.tsx`

#### a. Purpose

This is the **heart of the frontend** — the interactive component where users type their search query, submit it, and see results. It handles form state, makes the HTTP request to the FastAPI backend, parses the response, and renders the result cards.

#### b. Imports

```typescript
"use client";
```

This is a **Next.js directive** (not a regular import). It tells Next.js that this component runs in the **browser** (client-side), not on the server. This is necessary because the component uses `useState` (browser-only interactive state). Without it, Next.js would try to render it on the server and crash when it hits `useState`.

---

```typescript
import { useState } from "react";
```

**`useState`** is a React **hook** — a function that lets you add state (remembered values) to a component. When state changes, React re-renders the component to show the new values.

---

```typescript
import type { SearchHit, SearchResponse } from "@/lib/types";
```

Imports the TypeScript types we defined earlier, so the component knows the shape of the API response.

#### c. Line-by-line walkthrough

```typescript
function apiBase(): string {
  const raw =
    process.env.NEXT_PUBLIC_SEARCH_API_URL ?? "http://127.0.0.1:8000";
  return raw.replace(/\/$/, "");
}
```

A helper function that returns the API's base URL.

- **`process.env.NEXT_PUBLIC_SEARCH_API_URL`** reads an **environment variable**. In Next.js, variables prefixed with `NEXT_PUBLIC_` are exposed to the browser. This is where you configure the API URL for production.
- **`??`** is the **nullish coalescing operator** — if the left side is `null` or `undefined`, use the right side. Different from `||`, which also triggers on empty strings.
- **`.replace(/\/$/, "")`** removes a trailing slash if present. The regex `/\/$/` matches a `/` at the end (`$`) of the string. Without this, you might end up with `http://localhost:8000//search` (double slash).

---

```typescript
export default function SearchPanel() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchHit[]>([]);
```

Five pieces of state:

- **`query`** — what the user has typed.
- **`k`** — how many results to request.
- **`loading`** — whether a request is in flight.
- **`error`** — an error message, or `null` if there is none.
- **`results`** — the list of search hits from the API.

Each `useState` call returns a pair: the current value and a setter function. The argument to `useState` is the initial value. `useState<string | null>(null)` is the generic syntax — it tells TypeScript that `error` can be a `string` or `null`.

---

```typescript
  async function onSearch(e: React.FormEvent) {
    e.preventDefault();
```

This function runs when the user submits the form.

- **`async`** marks the function as asynchronous — it can use `await` inside to wait for Promises.
- **`e: React.FormEvent`** — the event object passed to form submission handlers.
- **`e.preventDefault()`** stops the browser's default form behavior (which would refresh the entire page). In a single-page app, you handle submissions with JavaScript instead.

---

```typescript
    setError(null);
    setResults([]);
    const q = query.trim();
    if (!q) {
      setError("Enter a search query.");
      return;
    }
    setLoading(true);
```

Resets error and results, trims the query, validates it, and sets loading state to true (which disables the button and shows "Searching...").

---

```typescript
    try {
      const res = await fetch(`${apiBase()}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, k }),
      });
```

**`fetch`** is the browser's built-in function for making HTTP requests.

- **`method: "POST"`** — we are sending data, not just requesting it.
- **`headers: { "Content-Type": "application/json" }`** tells the server the body is JSON.
- **`JSON.stringify({ query: q, k })`** converts the JavaScript object to a JSON string. Note that `{ query: q, k }` is shorthand for `{ query: q, k: k }` — when the key name matches the variable name, you can omit the colon.
- **`await`** pauses this function until the HTTP response comes back.

---

```typescript
      const text = await res.text();
      if (!res.ok) {
        let detail = text;
        try {
          const j = JSON.parse(text) as { detail?: string | unknown };
          if (typeof j.detail === "string") detail = j.detail;
        } catch {
          /* keep raw */
        }
        throw new Error(detail || `HTTP ${res.status}`);
      }
```

Error handling:

- First, read the response body as raw text.
- **`res.ok`** is `true` if the HTTP status is 200-299.
- If not OK, try to parse the body as JSON and extract a `detail` field (FastAPI puts error messages there).
- If that fails (maybe the response is not JSON), use the raw text.
- **`throw new Error(...)`** jumps to the `catch` block below.

---

```typescript
      const data = JSON.parse(text) as SearchResponse;
      setResults(data.results ?? []);
```

Parses the successful JSON response and updates the results state. `data.results ?? []` uses nullish coalescing to default to an empty array.

---

```typescript
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }
```

- **`catch`** handles any error thrown in the `try` block (network failure, parse error, our thrown error, etc.).
- **`finally`** runs no matter what — success or error — and sets loading back to false.

---

The JSX return is the component's visual output. Let's walk through the key sections:

```typescript
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-10">
```

A container div. The Tailwind classes:
- `mx-auto` — center horizontally.
- `max-w-3xl` — maximum width of about 768px.
- `flex flex-col gap-8` — vertical flex layout with spacing between children.
- `px-4 py-10` — padding: 1rem horizontal, 2.5rem vertical.

---

```typescript
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Florida MLS semantic search
        </h1>
        <p className="text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
          Search by meaning over listing descriptions. Powered by a FastAPI
          backend (embeddings + Chroma). Set{" "}
          <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">
            NEXT_PUBLIC_SEARCH_API_URL
          </code>{" "}
          for production.
        </p>
      </header>
```

The page header with title and description. `{" "}` is a JSX trick to preserve a space character — JSX collapses whitespace, so you need this for spaces between inline elements.

---

```typescript
      <form onSubmit={onSearch} className="flex flex-col gap-4">
```

A `<form>` element. **`onSubmit={onSearch}`** tells React to call `onSearch` when the form is submitted (either by clicking the button or pressing Enter).

---

```typescript
        <label className="flex flex-col gap-1 text-sm font-medium text-zinc-800 dark:text-zinc-200">
          Query
          <textarea
            className="min-h-[88px] rounded-lg border border-zinc-300 bg-white px-3 py-2 text-base text-zinc-900 shadow-sm outline-none focus:border-zinc-500 dark:border-zinc-600 dark:bg-zinc-950 dark:text-zinc-100"
            placeholder="e.g. screened pool renovated kitchen gated community"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
        </label>
```

A labeled `<textarea>`:
- **`value={query}`** — a **controlled component**: React controls the value, and the displayed text always matches the `query` state variable.
- **`onChange={(e) => setQuery(e.target.value)}`** — whenever the user types, update the `query` state with the new text. `e.target.value` is the current text in the textarea.
- **`disabled={loading}`** — disables the input while a search is in progress.
- **`min-h-[88px]`** — Tailwind's arbitrary value syntax: sets a minimum height of exactly 88 pixels.

---

```typescript
        <label className="flex max-w-xs flex-col gap-1 text-sm font-medium text-zinc-800 dark:text-zinc-200">
          Results (k)
          <input
            type="number"
            min={1}
            max={20}
            className="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-zinc-900 dark:border-zinc-600 dark:bg-zinc-950 dark:text-zinc-100"
            value={k}
            onChange={(e) => setK(Number(e.target.value) || 5)}
            disabled={loading}
          />
        </label>
```

A number input for `k`. `Number(e.target.value) || 5` converts the string to a number, and if conversion fails (resulting in `NaN`, which is falsy), defaults to 5.

---

```typescript
        <button
          type="submit"
          disabled={loading}
          className="w-fit rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
        >
          {loading ? "Searching…" : "Search"}
        </button>
```

The submit button. It changes text to "Searching..." while loading and becomes semi-transparent (`disabled:opacity-50`).

---

```typescript
      {error ? (
        <p
          className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200"
          role="alert"
        >
          {error}
        </p>
      ) : null}
```

**Conditional rendering:** If `error` is truthy (not null), render a red error box. If null, render nothing. `role="alert"` is an accessibility attribute that tells screen readers to announce this content immediately.

---

```typescript
      {results.length > 0 ? (
        <ol className="flex flex-col gap-6">
          {results.map((hit, i) => (
            <li
              key={hit.id}
              ...
            >
```

If there are results, render an ordered list. **`results.map((hit, i) => ...)`** loops through the results array and returns a JSX element for each one. **`key={hit.id}`** is required by React — it is a unique identifier that helps React efficiently update the list when items change.

---

```typescript
              <div className="mb-2 flex flex-wrap items-baseline gap-2 text-sm text-zinc-500 dark:text-zinc-400">
                <span className="font-semibold text-zinc-900 dark:text-zinc-100">
                  {i + 1}. {hit.id}
                </span>
                {hit.similarity != null && hit.distance != null ? (
                  <span>
                    similarity ≈ {hit.similarity.toFixed(3)} (distance{" "}
                    {hit.distance.toFixed(4)})
                  </span>
                ) : null}
              </div>
```

Displays the result rank (1-based, hence `i + 1`), the ID, and the similarity/distance if available. `.toFixed(3)` formats a number to 3 decimal places.

---

```typescript
              <dl className="mb-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
                <div>
                  <dt className="text-zinc-500 dark:text-zinc-400">Sold</dt>
                  <dd className="font-medium text-zinc-900 dark:text-zinc-100">
                    {String(hit.metadata.lastSoldPrice ?? "—")}
                  </dd>
                </div>
                ...
              </dl>
```

A **description list** (`<dl>`) with definition terms (`<dt>`) and definitions (`<dd>`). It uses a CSS grid layout: 2 columns on small screens (`grid-cols-2`), 4 columns on screens wider than `sm` (640px) — `sm:grid-cols-4`. This is **responsive design** via Tailwind breakpoint prefixes.

---

```typescript
              <p className="text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
                {hit.text.length > 1200 ? `${hit.text.slice(0, 1200)}…` : hit.text}
              </p>
```

Displays the listing text, truncated to 1200 characters if needed. `.slice(0, 1200)` extracts the first 1200 characters.

#### d. How this file connects to the rest

- **Imports:** `react` (useState), `@/lib/types` (SearchHit, SearchResponse).
- **Imported by:** `page.tsx`.
- **Communicates with:** The FastAPI backend via `fetch()` to `/search`.

---

### 5.12 `web/next.config.ts`

#### a. Purpose

This file holds Next.js configuration options. Currently it is empty (the default), but you could add rewrites, redirects, image optimization settings, environment variables, and more.

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
};

export default nextConfig;
```

It imports the `NextConfig` type for autocompletion, creates an empty config object, and exports it.

---

### 5.13 `web/postcss.config.mjs`

#### a. Purpose

**PostCSS** is a tool that transforms CSS using plugins. This config file tells PostCSS to use the `@tailwindcss/postcss` plugin, which is how Tailwind CSS v4 processes your `@import "tailwindcss"` directive and generates all the utility classes.

```javascript
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
```

The `.mjs` extension means this is an **ES module** (uses `export default` instead of `module.exports`). The empty `{}` means "use default settings for the plugin."

**Why no tailwind.config.ts?** Tailwind CSS v4 moved away from a separate config file. Configuration now happens through CSS directives (like `@theme inline` in `globals.css`) and the PostCSS plugin.

---

### 5.14 `web/tsconfig.json`

#### a. Purpose

This file configures the **TypeScript compiler** — the tool that checks your `.ts` and `.tsx` files for type errors and decides what JavaScript syntax they compile to.

#### b. Line-by-line walkthrough

```json
{
  "compilerOptions": {
    "target": "ES2017",
```

**`target`** — the version of JavaScript to compile TypeScript down to. `ES2017` supports `async/await` natively. Older targets would transform `async/await` into complex callback code.

---

```json
    "lib": ["dom", "dom.iterable", "esnext"],
```

**`lib`** — which built-in type definitions to include. `dom` gives you types for browser APIs (`document`, `fetch`, `HTMLElement`). `esnext` includes the latest JavaScript features.

---

```json
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
```

- **`allowJs`** — lets you use `.js` files alongside `.ts` files.
- **`skipLibCheck`** — skips type-checking inside `node_modules` (speeds up compilation).
- **`strict`** — enables all strict type-checking rules (catches more bugs).
- **`noEmit`** — do not actually output compiled `.js` files. Next.js handles compilation itself; TypeScript is only used for type checking.

---

```json
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
```

- **`esModuleInterop`** — allows importing CommonJS modules (like `const x = require(...)`) using `import x from ...` syntax.
- **`module`** — the module system to use in output (`esnext` uses `import/export`).
- **`moduleResolution: "bundler"`** — how TypeScript finds imported modules. `"bundler"` mimics how bundlers like webpack/turbopack resolve imports.

---

```json
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "react-jsx",
    "incremental": true,
```

- **`resolveJsonModule`** — allows importing `.json` files.
- **`isolatedModules`** — each file must be independently compilable (required by some build tools).
- **`jsx: "react-jsx"`** — how JSX is transformed. `"react-jsx"` uses the modern transform (no need to `import React` at the top of every file).
- **`incremental`** — caches compilation results so subsequent builds are faster.

---

```json
    "plugins": [
      {
        "name": "next"
      }
    ],
```

Loads the Next.js TypeScript plugin, which provides better autocompletion for Next.js-specific features.

---

```json
    "paths": {
      "@/*": ["./*"]
    }
```

**Path alias:** When you write `import X from "@/components/SearchPanel"`, TypeScript resolves `@/` to the project root (`./`). This is the configuration behind the `@/` shortcut you saw in the import statements.

---

```json
  "include": [
    "next-env.d.ts",
    "**/*.ts",
    "**/*.tsx",
    ".next/types/**/*.ts",
    ".next/dev/types/**/*.ts",
    "**/*.mts"
  ],
  "exclude": ["node_modules"]
}
```

- **`include`** — which files TypeScript should check. `**/*.ts` means "all `.ts` files in all subdirectories."
- **`exclude`** — skip `node_modules/` (thousands of third-party files that are already compiled).

---

### 5.15 `web/.env.local.example`

#### a. Purpose

This is a **template** for the `.env.local` file. The actual `.env.local` is git-ignored (it may contain secrets or local-only settings). This example shows the developer what variables to set.

```bash
# Copy to .env.local for local dev. No trailing slash.

NEXT_PUBLIC_SEARCH_API_URL=http://127.0.0.1:8000
```

- Lines starting with `#` are comments.
- **`NEXT_PUBLIC_SEARCH_API_URL`** — the URL of the FastAPI server. The `NEXT_PUBLIC_` prefix is required by Next.js for the variable to be accessible in browser-side code. Without this prefix, the variable would only be available in server-side code.

To use it, copy this file:

```bash
cp .env.local.example .env.local
```

---

### 5.16 `web/package.json`

#### a. Purpose

This file defines the Node.js project: its name, version, scripts (commands you can run), runtime dependencies, and development dependencies.

#### b. Key sections

```json
{
  "name": "web",
  "version": "0.1.0",
  "private": true,
```

- **`"private": true`** prevents accidentally publishing this package to npm (the public JavaScript registry).

---

```json
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint"
  },
```

**Scripts** are shortcuts. Running `npm run dev` executes `next dev`. The scripts are:

- **`dev`** — starts the development server with hot-reloading (changes appear instantly).
- **`build`** — compiles the app for production (optimized, minified).
- **`start`** — runs the production build.
- **`lint`** — checks code quality with ESLint.

---

```json
  "dependencies": {
    "next": "16.2.2",
    "react": "19.2.4",
    "react-dom": "19.2.4"
  },
```

**Dependencies** — libraries needed at runtime (when the app is running).

---

```json
  "devDependencies": {
    "@tailwindcss/postcss": "^4",
    "@types/node": "^20",
    "@types/react": "^19",
    "@types/react-dom": "^19",
    "eslint": "^9",
    "eslint-config-next": "16.2.2",
    "tailwindcss": "^4",
    "typescript": "^5"
  }
}
```

**Dev dependencies** — libraries needed only during development (type checking, linting, CSS processing). They are not included in the production bundle. The `^4` syntax means "any version starting with 4.x.x" (compatible updates only).

---

## 6. How to Run the Project

### Step 1: Install Python dependencies

Open a terminal at the repo root (`machine_learning/`):

```bash
pip install -r semantic_search/requirements.txt
```

- **`pip`** is Python's package installer.
- **`-r`** flag means "read requirements from this file."

### Step 2: Build the vector index

```bash
python semantic_search/embed_listings.py --limit 200
```

- **`python`** — runs the Python interpreter.
- **`semantic_search/embed_listings.py`** — the path to the script.
- **`--limit 200`** — only process the first 200 rows (for a quick test). Remove this flag for the full dataset.

**Expected output:**

```
Loading: /Users/sedly/Desktop/machine_learning/Florida Real Estate Sold Properties.csv
Rows to embed: 200
Loading model: sentence-transformers/all-MiniLM-L6-v2
Encoding (this may take a few minutes on full data)…
Batches: 100%|████████████████████| 4/4 [00:02<00:00]
  Indexed 200/200
Done. Vector store: /Users/sedly/Desktop/machine_learning/semantic_search/chroma_data
Next: npm run dev (in semantic_search/web)
```

**Common errors:**

| Error | Cause | Fix |
|-------|-------|-----|
| `CSV not found: ...` | Wrong working directory | `cd` to `machine_learning/` before running |
| `Expected column 'sanitized_text' in CSV` | CSV does not have the expected column | Check the CSV has been processed (column may need creating) |
| `ModuleNotFoundError: No module named 'chromadb'` | Dependencies not installed | Run `pip install -r semantic_search/requirements.txt` |

### Step 3: Start the FastAPI server

```bash
uvicorn semantic_search.server:app --reload --host 0.0.0.0 --port 8000
```

- **`uvicorn`** — the ASGI server.
- **`semantic_search.server:app`** — "import the `app` object from the `semantic_search.server` module." The colon separates the module path from the variable name.
- **`--reload`** — watch for file changes and restart automatically (great for development, do NOT use in production).
- **`--host 0.0.0.0`** — listen on all network interfaces (not just localhost).
- **`--port 8000`** — listen on port 8000.

**Expected output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using WatchFiles
```

You can test it immediately:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok","ready":true,"chroma_dir":"..."}
```

### Step 4: Start the Next.js frontend

In a **new terminal** (keep the API running):

```bash
cd semantic_search/web
cp .env.local.example .env.local
npm install
npm run dev
```

- **`cp .env.local.example .env.local`** — creates your local environment file from the template.
- **`npm install`** — installs JavaScript dependencies.
- **`npm run dev`** — starts the Next.js development server.

**Expected output:**

```
  ▲ Next.js 16.2.2
  - Local:   http://localhost:3000
  ✓ Ready
```

Open [http://localhost:3000](http://localhost:3000) in your browser. Type a query like "waterfront condo with ocean view" and click Search.

### Step 5 (optional): Deploy the API

There is no Dockerfile in this repo unless you add one. Deploy FastAPI using your host’s workflow (Railway, Render, Fly.io, ECS, etc.). See `README.md` for environment variables (`ALLOW_ORIGINS`, `PORT`) and how `chroma_data/` is produced for production.

---

## 7. Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **API (Application Programming Interface)** | A set of defined rules and URLs that programs use to communicate with each other. In this project, FastAPI provides a `/search` URL that the Next.js frontend calls. |
| **Argument parser** | A tool (`argparse`) that reads command-line flags (like `--limit 200`) and makes them available in your code as variables. |
| **ASGI (Asynchronous Server Gateway Interface)** | A standard for Python web servers and frameworks to communicate asynchronously. Uvicorn is an ASGI server; FastAPI is an ASGI framework. |
| **Async / await** | A way to write code that can wait for slow operations (like network requests) without blocking the entire program. `async` marks a function as asynchronous; `await` pauses it until a result is ready. |
| **Base image** | In Docker, the pre-built starting point for your container (e.g., `python:3.12-slim-bookworm`). |
| **Boolean indexing** | Filtering rows of a DataFrame by passing a list of True/False values — only rows where the value is True are kept. |
| **Build context** | The directory Docker sends to its build engine. Files outside this directory cannot be `COPY`-ed into the image. |
| **Clamping** | Restricting a number to a range. `max(1, min(k, 20))` clamps `k` between 1 and 20. |
| **Collection (ChromaDB)** | A named group of vectors inside a ChromaDB database, similar to a table in a traditional database. |
| **Component (React)** | A reusable piece of UI defined as a function. Components can have props (inputs) and state (remembered values). |
| **Controlled component** | A React component where the displayed value is driven by state (via `value={state}` and `onChange`), rather than the DOM managing it. |
| **CORS (Cross-Origin Resource Sharing)** | A browser security mechanism. By default, a website at `localhost:3000` cannot call an API at `localhost:8000`. CORS headers tell the browser it is allowed. |
| **Cosine distance** | A measure of how different two vectors are based on the angle between them. For normalized vectors: 0 = identical direction, 1 = perpendicular, 2 = opposite. |
| **CSS custom property (CSS variable)** | A value defined with `--name: value` and used with `var(--name)`. Allows reusing values across your stylesheet. |
| **DataFrame** | The primary data structure in pandas — a table with named columns and numbered rows, like a spreadsheet. |
| **Decorator** | A function that wraps another function to add behavior. In Python, decorators use `@syntax` above the function definition (e.g., `@lru_cache`, `@app.get`). |
| **Destructuring** | Extracting values from objects or arrays into variables. `const { query } = props` pulls `query` out of `props`. |
| **Docker container** | A lightweight, isolated environment that packages your app with all its dependencies. |
| **Embedding** | A fixed-length list of numbers (a vector) that captures the meaning of a piece of text. Similar texts get similar embeddings. |
| **Endpoint** | A specific URL path on a server that handles a specific type of request (e.g., `/search` or `/health`). |
| **Environment variable** | A key-value setting that lives outside your code, in the operating system or deployment platform. Used for configuration that changes between environments (dev vs. production). |
| **f-string** | A Python string prefixed with `f` that can embed variables: `f"Hello {name}"`. |
| **Fail-fast** | A design pattern where the program checks for problems and stops immediately with a clear error message, rather than continuing and crashing later with a confusing one. |
| **Fetch** | The browser's built-in function for making HTTP requests (`fetch(url, options)`). |
| **Hook (React)** | A function (like `useState`, `useEffect`) that lets you use React features inside function components. |
| **HNSW (Hierarchical Navigable Small World)** | An algorithm for fast approximate nearest-neighbor search. ChromaDB uses it internally to find the closest vectors without checking every single one. |
| **HTTP status code** | A number indicating the result of an HTTP request. 200 = OK, 422 = validation error, 503 = service unavailable. |
| **JSX** | A syntax extension that lets you write HTML-like code inside JavaScript/TypeScript. Looks like HTML but is actually function calls. |
| **JSON (JavaScript Object Notation)** | A text format for structured data: `{"key": "value", "count": 42}`. The standard format for API communication. |
| **Layer caching (Docker)** | Docker caches each build step. If a step's inputs have not changed, Docker reuses the cached result instead of re-running it. |
| **Lazy initialization** | Delaying the creation of an expensive object until it is first needed. |
| **List comprehension** | A compact Python syntax for building lists: `[x*2 for x in range(5)]` produces `[0, 2, 4, 6, 8]`. |
| **Metadata** | Extra information attached to each stored vector (like price, sqft, ZIP code). Returned alongside search results. |
| **Middleware** | Code that runs on every HTTP request before it reaches the endpoint. Used for cross-cutting concerns like CORS, logging, or authentication. |
| **Module** | A single Python file that can be imported by other files. |
| **Nearest neighbor** | In vector search, the stored item(s) whose vectors are closest (smallest distance) to the query vector. |
| **Normalization (vector)** | Scaling a vector so its length (magnitude) equals 1. This ensures cosine distance measures direction similarity, not magnitude. |
| **Nullish coalescing (`??`)** | A JavaScript operator that returns the right side if the left is `null` or `undefined`. |
| **Package (Python)** | A directory with an `__init__.py` file, containing one or more modules. |
| **Path alias (`@/`)** | A shortcut configured in `tsconfig.json` so `@/components/X` resolves to `./components/X`. |
| **Pydantic model** | A class inheriting from `BaseModel` that validates and serializes data. FastAPI uses them for request/response schemas. |
| **REST API** | An architectural style for APIs where resources are accessed via HTTP methods (GET, POST, PUT, DELETE) and URLs. |
| **Responsive design** | Making a UI adapt to different screen sizes. Tailwind's breakpoint prefixes (`sm:`, `md:`, `lg:`) apply styles conditionally based on screen width. |
| **Route** | A mapping between a URL path and a function that handles requests to it. |
| **Semantic search** | Searching by meaning rather than exact keyword matching. Uses embeddings to compare the intent of a query to the intent of stored documents. |
| **Sentence transformer** | A neural network model fine-tuned to produce embeddings where semantically similar sentences have nearby vectors. |
| **Singleton** | An object that is created only once and reused everywhere (like the ChromaDB collection in `core.py`). |
| **State (React)** | Data that a component "remembers" between renders. When state changes, the component re-renders. |
| **Type alias** | A name you give to a type shape in TypeScript: `type SearchHit = { id: string; ... }`. |
| **Type hint** | An annotation in Python (like `x: int`) that tells tools and humans what type a variable should be. Does not affect runtime. |
| **Union type** | A type that can be one of several types: `string | null` means "a string or null." |
| **Unit vector** | A vector with a length of exactly 1. Created by dividing each component by the vector's magnitude. |
| **Utility class (Tailwind)** | A CSS class that does one thing: `bg-white` sets background to white, `text-sm` sets small text, `px-4` adds horizontal padding. |
| **Vector** | An ordered list of numbers (like `[0.12, -0.45, 0.89, ...]`). In this project, each vector has 384 numbers and represents the meaning of a text. |
| **Vector database** | A database optimized for storing and searching through vectors using distance metrics (cosine, Euclidean, etc.). |
| **Virtual environment** | An isolated Python installation that keeps packages for one project separate from all others on your machine. |

---

## 8. Next Steps for Learning

Here are things you could try to deepen your understanding. Each one is a small experiment, not a full project:

1. **Change the number of results.** In `core.py`, `MAX_K` is 20. Try changing it to 50 and see what happens in the UI. Then think about: why would someone want a limit at all? What happens if you request more results than there are items in the database?

2. **Try a different embedding model.** Run `embed_listings.py` with `--model sentence-transformers/all-mpnet-base-v2` (a more powerful but slower model that produces 768-dimensional vectors). Then search for the same queries and compare results. Connect this to the concept of the "vector space" discussed in `constants.py` — why must the query model match the ingest model?

3. **Add a new metadata field.** The CSV probably has columns that are not included in the metadata (like `city` or `address`). In `embed_listings.py`, add another entry to the `metadatas` dictionary. Then display it in `SearchPanel.tsx`. This exercises the full pipeline from ingestion to display.

4. **Add a filter to the search.** ChromaDB supports `where` clauses (like SQL's `WHERE`). Try modifying `search_listings()` in `core.py` to accept an optional `zip_filter` parameter and pass it to `collection.query(where={"zip": zip_filter})`. This connects to the idea of combining semantic search with traditional filtering.

5. **Look at the raw vectors.** After running `embed_listings.py`, try this in a Python shell:

```python
from semantic_search.core import get_collection
col = get_collection()
result = col.get(ids=["row_0"], include=["embeddings"])
print(len(result["embeddings"][0]))  # Should print 384
print(result["embeddings"][0][:10])  # First 10 numbers of the vector
```

This makes embeddings tangible — they are just lists of numbers.

6. **Connect to your earlier work.** If you have `linear_regression_multifeature.py` or other ML projects in the `machine_learning/` folder, notice the pattern: in linear regression you had feature vectors (sqft, beds, baths) that predicted a price. Here you have text vectors (384 dimensions) that measure similarity. Both are "turn a thing into numbers, then do math on those numbers." The difference is that embedding models learn which numbers to produce, while in linear regression you hand-picked the features.

7. **Explore the auto-generated API docs.** With the FastAPI server running, visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). FastAPI automatically generates an interactive documentation page where you can try API calls directly in the browser. Notice how the Pydantic models in `server.py` become the request/response schemas in the docs.
