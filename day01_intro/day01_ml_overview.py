# ==============================================================
# DAY 1 — MACHINE LEARNING OVERVIEW & PYTHON BASICS
# Course: ML Fundamentals | Week 1
# ==============================================================

# ---------------------------------------------------------------
# WHAT IS A "COMMENT" IN PYTHON?
# Lines that start with the '#' symbol are called COMMENTS.
# Python completely ignores them — they are just notes for humans.
# We'll use lots of comments to explain what the code does!
# ---------------------------------------------------------------


# ==============================================================
# SECTION 1: WHAT IS MACHINE LEARNING?
# ==============================================================

# Machine Learning (ML) is a way to teach computers to learn
# from data — instead of giving the computer exact instructions,
# we show it examples and let it figure out the pattern.

# REAL-WORLD EXAMPLES:
#   - Gmail filtering spam emails automatically
#   - Netflix recommending what to watch next
#   - Your phone recognising your face to unlock
#   - A bank detecting fraudulent credit card transactions

print("=" * 60)
print("WELCOME TO YOUR MACHINE LEARNING COURSE!")
print("=" * 60)

# 'print()' is a Python function that displays text on the screen.
# Anything inside the quotes gets printed.

print("\nWhat is Machine Learning?")
print("Teaching computers to learn from data, not rules.")


# ==============================================================
# SECTION 2: TYPES OF MACHINE LEARNING
# ==============================================================

print("\n" + "=" * 60)
print("TYPES OF MACHINE LEARNING")
print("=" * 60)

# --- TYPE 1: SUPERVISED LEARNING ---
# The computer learns from LABELED examples.
# (Like a student learning with an answer key!)
# Examples:
#   - Predicting house prices (we have past prices → label)
#   - Email spam detection (emails labeled spam/not-spam)

print("\n1. SUPERVISED LEARNING")
print("   - Learns from labeled data")
print("   - Examples: predicting prices, spam detection")

# --- TYPE 2: UNSUPERVISED LEARNING ---
# The computer finds hidden PATTERNS without labels.
# (Like grouping similar things without being told how!)
# Examples:
#   - Grouping customers by shopping habits (clustering)
#   - Compressing images (finding structure in data)

print("\n2. UNSUPERVISED LEARNING")
print("   - Finds patterns without labels")
print("   - Examples: customer segmentation, anomaly detection")

# --- TYPE 3: REINFORCEMENT LEARNING ---
# The computer learns by trial & error, getting REWARDS or PENALTIES.
# (Like training a dog with treats!)
# Examples:
#   - AlphaGo (Google AI that beats humans at chess/Go)
#   - Self-driving cars learning to navigate

print("\n3. REINFORCEMENT LEARNING")
print("   - Learns through rewards and penalties")
print("   - Examples: game AI, robotics, self-driving cars")


# ==============================================================
# SECTION 3: THE ML LIFECYCLE (THE BIG PICTURE)
# ==============================================================

print("\n" + "=" * 60)
print("THE MACHINE LEARNING LIFECYCLE")
print("=" * 60)

# Every ML project follows these steps — this course covers all of them!

steps = [
    "1. Define the Problem     — What question are we answering?",
    "2. Collect Data           — Gather raw information",
    "3. Clean & Preprocess     — Fix missing values, outliers, etc.",
    "4. Exploratory Analysis   — Understand the data (graphs, stats)",
    "5. Feature Engineering    — Create useful input columns",
    "6. Train a Model          — Let the algorithm learn",
    "7. Evaluate               — How accurate is the model?",
    "8. Deploy                 — Put the model into production",
    "9. Monitor                — Watch for performance changes",
]

# 'for' loop: repeats an action for each item in a list
# 'step' takes the value of each item one-by-one
for step in steps:
    print("  ", step)


# ==============================================================
# SECTION 4: PYTHON BASICS REFRESHER
# ==============================================================

print("\n" + "=" * 60)
print("PYTHON BASICS YOU NEED TO KNOW")
print("=" * 60)

# --- VARIABLES ---
# A variable is like a box that stores a value.
# You give it a name, then use = to put something inside.

name = "Alice"          # This is a STRING (text, always in quotes)
age = 25                # This is an INTEGER (whole number)
salary = 48500.50       # This is a FLOAT (decimal number)
is_member = True        # This is a BOOLEAN (only True or False)

print("\nVariables:")
print("Name:", name)
print("Age:", age)
print("Salary:", salary)
print("Is Member:", is_member)

# --- LISTS ---
# A list stores MULTIPLE values in one variable.
# Use square brackets [ ] and separate items with commas.

fruits = ["apple", "banana", "cherry", "date"]
print("\nList example:", fruits)
print("First item:", fruits[0])   # Lists start counting from 0!
print("Last item:", fruits[-1])   # -1 gives the last item

# --- DICTIONARIES ---
# A dictionary stores KEY-VALUE pairs (like a real dictionary: word → meaning)

student = {
    "name": "Bob",
    "age": 22,
    "grade": "A",
}
print("\nDictionary example:", student)
print("Student name:", student["name"])   # Access by key

# --- FUNCTIONS ---
# A function is a reusable block of code.
# 'def' means "define a new function"

def greet(person_name):
    """This function says hello to someone."""
    # 'return' sends a value back to whoever called the function
    return "Hello, " + person_name + "! Welcome to ML class."

# Call the function by writing its name with ()
message = greet("Charlie")
print("\nFunction output:", message)


# ==============================================================
# SECTION 5: PYTHON LIBRARIES WE'LL USE
# ==============================================================

print("\n" + "=" * 60)
print("KEY PYTHON LIBRARIES FOR ML")
print("=" * 60)

# Libraries are collections of pre-written code that we can use.
# Instead of writing everything from scratch, we 'import' libraries.

libraries = {
    "numpy":      "Fast math and array operations",
    "pandas":     "Working with tables of data (like Excel in Python)",
    "matplotlib": "Creating charts and graphs",
    "seaborn":    "Beautiful statistical visualizations",
    "sklearn":    "Machine learning algorithms (scikit-learn)",
}

for lib, description in libraries.items():
    print(f"  {lib:<12} → {description}")

# f-string (f"...") lets you put variable values directly in text
# {lib:<12} prints lib left-aligned in a space of 12 characters


# ==============================================================
# SECTION 6: QUICK DEMO — IMPORTING AND USING A LIBRARY
# ==============================================================

print("\n" + "=" * 60)
print("DEMO: LOADING OUR DATASET")
print("=" * 60)

# 'import pandas as pd' means: bring in the pandas library,
# and let us call it 'pd' for short (saves typing!)
import pandas as pd

# Load the raw CSV file into a pandas DataFrame.
# A DataFrame is like a spreadsheet (rows and columns).
# '..' goes up one folder, then we go into 'data/'
df = pd.read_csv("../data/raw_data.csv")

# Show basic info about our dataset
print("\nDataset Shape (rows, columns):", df.shape)
# .shape tells us how many rows and columns exist

print("\nColumn Names:")
print(df.columns.tolist())   # .columns gives column names, .tolist() makes it a list

print("\nFirst 5 rows of data:")
print(df.head())   # .head() shows the first 5 rows by default


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("DAY 1 COMPLETE! KEY TAKEAWAYS:")
print("=" * 60)
print("✓ ML teaches computers to learn from data")
print("✓ Three main types: Supervised, Unsupervised, Reinforcement")
print("✓ ML lifecycle: collect → clean → analyse → model → deploy")
print("✓ Python variables, lists, dicts, and functions")
print("✓ We will use pandas, matplotlib, seaborn, and sklearn")
print("\nTomorrow: Data Sources & Collection!")
