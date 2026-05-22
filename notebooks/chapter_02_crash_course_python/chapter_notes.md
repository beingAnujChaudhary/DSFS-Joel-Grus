# Chapter 2 — A Crash Course in Python 
## Chapter overview

This chapter is a rapid, focused tour of Python features essential for data science and machine learning. Rather than teaching Python from scratch, it highlights the patterns, data structures, and idioms you'll use repeatedly throughout the book. The emphasis is on writing **Pythonic code** — clean, readable, efficient, and production-ready.

Instead of covering every Python feature, this chapter focuses on the parts most relevant to data science and AI engineering, including:

- Variables and data types
- Core data structures (lists, tuples, dictionaries, sets)
- Control flow and truthiness
- Functions and lambda expressions
- List comprehensions and generators
- Functional programming tools
- Exception handling
- Classes and object-oriented basics

The goal is not mastering Python completely, but becoming productive enough to implement data science concepts and build AI systems.

---

## Learning objectives

After completing this chapter, I should be able to:

- Use Python's core data structures effectively (lists, tuples, dicts, sets)
- Write concise transformations using list comprehensions and `zip`
- Leverage `defaultdict` and `Counter` for counting and grouping
- Understand generators and iterators for memory-efficient processing
- Apply functional tools (`map`, `filter`, `partial`, `enumerate`) correctly
- Write flexible functions using `*args` and `**kwargs`
- Use Python's truthiness rules to write cleaner conditionals
- Handle errors gracefully with exceptions
- Build structured code using classes

---

## Key terms

| Term | Definition |
|------|------------|
| Variable | Named storage for data |
| Function | Reusable block of code |
| Argument | Value passed to a function |
| List | Ordered, mutable collection |
| Tuple | Ordered, immutable collection |
| Dictionary | Key-value mapping |
| Set | Unordered collection of unique items |
| List comprehension | Compact syntax for creating lists |
| Generator | Lazy iterator yielding values one at a time |
| Iterator | Object that produces values sequentially |
| Lambda | Anonymous inline function |
| Class | Blueprint for creating objects |
| Exception | Runtime error that can be handled |
| Functional programming | Programming style using pure functions |
| Object-oriented programming (OOP) | Programming style using objects |

---

## Mental model

**Python is a toolbox.** Different tools solve different problems:

```txt
Variables         → store information
Functions         → reusable logic
Lists             → ordered sequences
Tuples            → immutable groups
Dictionaries      → key-value lookups
Sets              → unique collections
Loops             → repetition
List comprehensions → clean transformations
Generators        → memory-efficient iteration
Classes           → structured objects
Exceptions        → graceful error handling
Lambda functions  → quick, single-use logic
```

**The goal of this chapter:**
> Write clean, readable, reusable code that solves real problems.

---

## Key concepts

### 1. Variables

Variables store values.

**Example:**
```python
x = 10
name = "Anuj"
```

**Dynamically typed:**
```python
x = 10      # x is an integer
x = "hello" # now x is a string
```

The same variable can hold different types — flexible but requires discipline.

---

### 2. Numbers

Python supports:
- **Integers** (`int`): whole numbers
- **Floats** (`float`): decimal numbers

**Examples:**
```python
x = 10
pi = 3.14159
```

**Operations:**
```python
+   # addition
-   # subtraction
*   # multiplication
/   # float division (returns float)
//  # integer division (floor)
%   # modulo (remainder)
**  # exponentiation
```

**Examples:**
```python
10 / 3   # 3.333...
10 // 3  # 3
10 % 3   # 1
2 ** 5   # 32
```

---

### 3. Strings

Strings represent text.

**Examples:**
```python
name = "Anuj"
message = 'Hello'
multi_line = """This spans
multiple lines"""
```

**Useful methods:**
```python
len(name)          # length: 4
name.upper()       # "ANUJ"
name.lower()       # "anuj"
name.split("u")    # ['An', 'j']
" - ".join(["a", "b"])  # "a - b"
```

**String formatting (f-strings):**
```python
name = "Anuj"
age = 25
f"Hello {name}, you are {age} years old"
# "Hello Anuj, you are 25 years old"
```

**Why it matters:** Text data is everywhere in ML — feature names, labels, prompts, API responses.

---

### 4. Lists

Lists are **ordered, mutable** collections.

**Example:**
```python
numbers = [1, 2, 3]
mixed = [1, "hello", 3.14, True]
empty = []
```

**Common operations:**
```python
numbers.append(4)        # [1, 2, 3, 4]
numbers.pop()            # removes last, returns 4
numbers.insert(1, 99)    # [1, 99, 2, 3]
numbers.sort()           # sorts in place
len(numbers)             # length
```

**Indexing and slicing:**
```python
numbers[0]      # first element
numbers[-1]     # last element
numbers[:3]     # first 3 elements
numbers[1:]     # elements from index 1
numbers[::2]    # every 2nd element
```

**Important:** Lists are mutable — they can be changed after creation.

---

### 5. Tuples

Tuples are **ordered but immutable** (cannot be changed).

**Example:**
```python
point = (10, 20)
coordinates = (x, y, z)
single_item = (42,)  # comma required!
```

**Why use tuples:**
- Values that shouldn't change (e.g., RGB colors, fixed coordinates)
- More memory-efficient than lists
- Can be used as dictionary keys (lists cannot)

**Tuple unpacking:**
```python
x, y = (10, 20)  # x=10, y=20
```

**When to use tuple vs list:**
| Use Case | Choice | Reason |
|----------|--------|--------|
| Coordinates | Tuple | Fixed, shouldn't change |
| Daily temperatures | List | Changes daily |
| Function returns multiple values | Tuple | Standard pattern |
| Collection to modify frequently | List | Mutable |

---

### 6. Dictionaries

Dictionaries store **key-value pairs** with fast lookup.

**Example:**
```python
student = {
    "name": "Anuj",
    "age": 20,
    "courses": ["Math", "CS"]
}
```

**Access and methods:**
```python
student["name"]              # "Anuj"
student.get("grade", "N/A")  # "N/A" (safe access)

student.keys()    # dict_keys(['name', 'age', 'courses'])
student.values()  # dict_values(['Anuj', 20, ['Math', 'CS']])
student.items()   # key-value pairs

# Adding/updating
student["grade"] = "A"
```

**Why it matters:** Dictionaries are everywhere in ML:
- Configuration parameters
- Feature dictionaries
- Model hyperparameters
- API request/response data

---

### 7. Specialized Dictionaries: `defaultdict` and `Counter`

These are **production-grade tools** that reduce boilerplate.

**`defaultdict`** — Automatically handles missing keys:

```python
from collections import defaultdict

# Without defaultdict (verbose)
groups = {}
for item in data:
    if item not in groups:
        groups[item] = []
    groups[item].append(value)

# With defaultdict (clean)
groups = defaultdict(list)
for item in data:
    groups[item].append(value)  # Works even if key missing!
```

**`Counter`** — Perfect for counting frequencies:

```python
from collections import Counter

words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
counts = Counter(words)
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})

counts.most_common(2)  # [('apple', 3), ('banana', 2)]
```

**Why they matter:** These replace verbose `if key in dict` checks and manual counting loops — foundational for data aggregation and vocabulary building in NLP.

---

### 8. Sets

Sets store **unique, unordered** values with fast membership testing.

**Example:**
```python
unique_numbers = {1, 2, 3}
unique_from_list = set([1, 2, 2, 3, 3])  # {1, 2, 3}
```

**Operations:**
```python
2 in unique_numbers        # True (O(1) lookup)
unique_numbers.add(4)      # {1, 2, 3, 4}
unique_numbers.remove(2)   # {1, 3, 4}

# Set operations
a = {1, 2, 3}
b = {3, 4, 5}
a | b  # union: {1, 2, 3, 4, 5}
a & b  # intersection: {3}
a - b  # difference: {1, 2}
```

**Use cases:**
- Removing duplicates
- Fast membership checking
- Finding common elements between collections

---

### 9. Conditionals

Used for decision making in code.

**Basic syntax:**
```python
if x > 10:
    print("Large")
elif x == 10:
    print("Equal")
else:
    print("Small")
```

**Python's truthiness** — Values that are considered `False`:
```python
False, None, 0, 0.0, "", [], {}, set()
```

**Cleaner conditionals with truthiness:**
```python
# Instead of:
if len(data) > 0:
    process(data)

# Use:
if data:  # Empty collections are falsy
    process(data)

# Instead of:
if value is not None and value > 0:

# Use:
if value:  # None and 0 are falsy, but be careful with 0
```

---

### 10. Loops

#### `for` loops — Iterate over sequences:

```python
# Over a list
for item in [1, 2, 3]:
    print(item)

# With range
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

# With enumerate (get index + value)
for i, value in enumerate(["a", "b", "c"]):
    print(f"Index {i}: {value}")
```

#### `while` loops — Continue while condition is true:

```python
count = 0
while count < 5:
    print(count)
    count += 1  # Increment to avoid infinite loop
```

#### `for/else` — Run code if loop completes without `break`:

```python
for item in items:
    if condition(item):
        break
else:
    print("No break occurred — item not found")
```

---

### 11. Functions

Functions make code **reusable**, **testable**, and **organised**.

**Basic function:**
```python
def square(x):
    """Return the square of x."""
    return x * x

result = square(5)  # 25
```

**Default arguments:**
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Anuj")           # "Hello, Anuj!"
greet("Anuj", "Hi")     # "Hi, Anuj!"
```

**Flexible arguments with `*args` and `**kwargs`:**

```python
# *args: arbitrary positional arguments (tuple)
def sum_all(*args):
    return sum(args)

sum_all(1, 2, 3, 4)  # 10

# **kwargs: arbitrary keyword arguments (dict)
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Anuj", age=25, role="Engineer")
# name: Anuj
# age: 25
# role: Engineer
```

**Why `*args`/`**kwargs` matter:** Essential for building wrappers, decorators, and flexible ML model APIs where you don't know all parameters in advance.

---

### 12. Lambda Functions

Small, anonymous functions for quick transformations.

**Syntax:** `lambda arguments: expression`

```python
# Regular function
def square(x):
    return x * x

# Lambda equivalent
square = lambda x: x * x

# Used inline (most common)
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
# [1, 4, 9, 16, 25]

# With sorted (custom key)
students = [("Anuj", 85), ("Bob", 75), ("Charlie", 95)]
sorted(students, key=lambda x: x[1])  # Sort by score
# [('Bob', 75), ('Anuj', 85), ('Charlie', 95)]
```

**Best practice:** Use lambdas for simple operations. Use `def` for complex logic or reusable functions.

---

### 13. List Comprehensions

The **Pythonic** way to create lists — cleaner and faster than manual loops.

**Instead of:**
```python
squares = []
for x in range(5):
    squares.append(x * x)
```

**Use:**
```python
squares = [x * x for x in range(5)]  # [0, 1, 4, 9, 16]
```

**With condition (filtering):**
```python
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
```

**With transformation:**
```python
words = ["hello", "world", "python"]
upper_words = [word.upper() for word in words]
# ['HELLO', 'WORLD', 'PYTHON']
```

**Nested comprehensions (multiple loops):**
```python
pairs = [(x, y) for x in range(3) for y in range(3)]
# [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
```

**Why it matters:** List comprehensions are:
- **More readable** — intent is clear
- **Faster** — optimized C-level execution
- **Standard** — widely used in data science

---

### 14. `zip` — Pair Sequences Together

`zip` combines multiple sequences element-wise.

```python
names = ["Alice", "Bob", "Charlie"]
scores = [90, 85, 92]
paired = list(zip(names, scores))
# [('Alice', 90), ('Bob', 85), ('Charlie', 92)]

# Unzipping
unzipped_names, unzipped_scores = zip(*paired)

# With three sequences
features = [[1,2], [3,4], [5,6]]
labels = [0, 1, 0]
for feat, label in zip(features, labels):
    print(f"Features: {feat}, Label: {label}")
```

**Why it matters:** Essential for pairing features with labels, iterating over multiple collections simultaneously, and building training batches.

---

### 15. Generators and Iterators

Generators produce values **lazily** — one at a time — instead of loading everything into memory.

**Generator function (using `yield`):**
```python
def lazy_range(n):
    i = 0
    while i < n:
        yield i  # Produces one value, then pauses
        i += 1

for num in lazy_range(1000000):  # Memory efficient!
    print(num)
    if num > 5:
        break
```

**Generator expression (parentheses, not brackets):**
```python
# List comprehension (creates list immediately — memory heavy)
squares_list = [x**2 for x in range(1000000)]  # Allocates memory

# Generator expression (lazy — memory efficient)
squares_gen = (x**2 for x in range(1000000))   # No memory allocated yet

# Sum without storing
total = sum(x**2 for x in range(1000000))
```

**Key traits:**
- Memory-efficient for large datasets
- Can only be iterated **once**
- Perfect for streaming data pipelines

**When to use generators vs lists:**
| Scenario | Use | Reason |
|----------|-----|--------|
| Small dataset (< 10K items) | List | Simple and readable |
| Large dataset | Generator | Memory efficiency |
| Need multiple passes | List | Generators can't rewind |
| Streaming/real-time data | Generator | Process as data arrives |
| Training data loader | Generator | Batch loading pattern |

---

### 16. Functional Programming Tools

Python supports functional programming patterns that enable clean data pipelines.

| Tool | Purpose | Example |
|------|---------|---------|
| `enumerate()` | Index + value in loops | `for i, val in enumerate(data)` |
| `map()` | Apply function to every element | `list(map(str.upper, words))` |
| `filter()` | Keep items matching condition | `list(filter(lambda x: x > 0, numbers))` |
| `partial()` | Fix some function arguments | `add_5 = partial(add, 5)` |
| `sorted()` | Sort with custom key | `sorted(data, key=lambda x: x['score'])` |

**Example: Clean preprocessing pipeline**
```python
from functools import partial

def process_text(text, lowercase=True, remove_punct=False):
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = text.replace(".", "").replace(",", "")
    return text

# Create specialized versions
process_lower = partial(process_text, lowercase=True, remove_punct=False)
process_clean = partial(process_text, lowercase=True, remove_punct=True)

texts = ["Hello, World!", "Python. Rocks!", "Data Science"]
cleaned = list(map(process_clean, texts))
# ['hello world', 'python rocks', 'data science']
```

---

### 17. Exception Handling

Handle errors gracefully instead of crashing.

**Basic pattern:**
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
    result = None
```

**Multiple exception types:**
```python
try:
    value = int(input("Enter a number: "))
    result = 100 / value
except ValueError:
    print("That's not a valid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
```

**`else` and `finally`:**
```python
try:
    file = open("data.txt")
    content = file.read()
except FileNotFoundError:
    print("File not found")
else:
    print("File read successfully")  # Runs if no exception
finally:
    file.close()  # Always runs (cleanup)
```

**Why it matters:** Production systems need graceful failure — don't let a single bad input crash your entire pipeline.

---

### 18. Classes (Object-Oriented Programming)

Classes group **data** (attributes) and **behavior** (methods) together.

**Basic class:**
```python
class Person:
    def __init__(self, name, age):
        """Constructor — called when object is created."""
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, my name is {self.name}"

    def have_birthday(self):
        self.age += 1
        return f"Happy birthday! I'm now {self.age}"

# Usage
anuj = Person("Anuj", 25)
print(anuj.greet())           # "Hello, my name is Anuj"
anuj.have_birthday()          # Age becomes 26
```

**Inheritance — extending existing classes:**
```python
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)  # Call parent constructor
        self.student_id = student_id

    def greet(self):  # Override parent method
        return f"Hi, I'm student {self.student_id}"

student = Student("Bob", 20, "S123")
print(student.greet())  # "Hi, I'm student S123"
```

**Why classes matter in AI/ML:**
- **Modelling real-world entities** (Dataset, Model, Pipeline)
- **Encapsulation** — hide complex internals
- **Code organisation** — large projects need structure
- **Reusable components** — inherit and extend

**Example: Simple ML Model Class**
```python
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None

    def train(self, X, y):
        """Train the model on data."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        # Training logic here

    def predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights)

    def score(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

---

## Important takeaways

### 1. Pythonic > Verbose
- Prefer `defaultdict` over manual `if key not in dict`
- Prefer list comprehensions over `for` loops with `.append()`
- Prefer `zip` and `enumerate` over manual index tracking

### 2. Memory efficiency matters
- Generators and iterators prevent `MemoryError` on large datasets
- Lazy evaluation is a core pattern in production data pipelines

### 3. Flexibility via `*args` and `**kwargs`
- Enables building wrappers, decorators, and higher-order functions
- Essential for passing hyperparameters and dynamic configurations

### 4. Truthiness reduces boilerplate
- `if data:` replaces `if data is not None and len(data) > 0:`
- Cleaner, more readable conditionals

### 5. Readability is a feature
Python prioritises **readable and simple code** over clever tricks.

### 6. Lists and dictionaries are foundational
Most data science workflows rely heavily on:
- Lists (sequences)
- Dictionaries (lookups)
- Iteration (processing)

### 7. Functions improve code quality
Avoid repeating logic. Use functions.

### 8. List comprehensions are powerful
Cleaner than many loops. Widely used in data science.

### 9. Generators save memory
Critical when working with large datasets.

### 10. Error handling matters
Good engineers anticipate failures.

---

## Connections to AI/ML engineering

This chapter connects directly to AI/ML engineering:

### Data processing
Using loops, functions, dictionaries, and list comprehensions to clean and transform data.

### ML pipelines
Generators help with large datasets, streaming, and batch processing.

### Feature engineering
`defaultdict` and `Counter` for vocabulary building and frequency analysis.

### Model training
Generators for memory-efficient training loops and data loaders.

### APIs and production systems
Classes for system organisation, reusable components, and structured applications.

### Flexible APIs
`*args`/`**kwargs` for hyperparameter passing and decorator patterns.

### Robust systems
Exception handling for graceful failure in production.

**Strong Python fundamentals now will make debugging, optimizing, and scaling ML systems significantly easier later.**

---

## Questions / confusions

Things to revisit:

- [ ] When should I prefer tuples vs lists? *(Answer: tuples for fixed data, lists for changing data)*
- [ ] When are generators better than lists? *(Answer: large datasets, streaming, memory constraints)*
- [ ] When should I use classes instead of functions? *(Answer: multiple related functions sharing state, complex objects)*
- [ ] How much OOP is used in ML engineering? *(Answer: heavily — datasets, models, pipelines, trainers)*
- [ ] How do `*args`/`**kwargs` interact with modern type hints?
- [ ] What are the performance trade-offs of `map`/`filter` vs list comprehensions in Python 3?
- [ ] How does `defaultdict` handle nested structures efficiently?

---

## Personal reflections

### What surprised me?
Python is simpler than expected but has many powerful shortcuts. How much of data preprocessing can be expressed cleanly with `Counter`, `defaultdict`, and list comprehensions.

### What stood out?
Python's truthiness rules and `try/except` patterns drastically reduce boilerplate. List comprehensions and dictionaries feel extremely useful. Writing "Pythonic" code isn't just style — it's about leveraging the language's design for clarity and performance.

### How does this connect to my roadmap?
Python fluency is essential for: Data science, machine learning, MLOps, APIs, LLM systems, production AI engineering. This chapter is the foundation for everything that follows. Understanding iterators, functional patterns, and flexible function signatures will make it easier to work with pandas, scikit-learn pipelines, and custom training loops without fighting the language.

Strong Python fundamentals will reduce friction later.

---

## Chapter summary (in my own words)

Chapter 2 is a practical crash course in the Python features that matter most for data science. It moves quickly through syntax to focus on idiomatic patterns: `defaultdict` and `Counter` for aggregation, list comprehensions and `zip` for clean transformations, generators for memory efficiency, and `*args`/`**kwargs` for flexibility.

I learned about variables, functions, loops, lists, dictionaries, sets, list comprehensions, generators, exception handling, and classes. The biggest lesson is that Python values readability and simplicity, while still offering powerful features for writing efficient and reusable code.

The chapter reinforces that writing Pythonic code isn't about clever tricks — it's about choosing the right tool for readability, performance, and maintainability. These patterns will appear repeatedly in every subsequent chapter and in real-world ML engineering.

---

## Quick reference cheatsheet

```python
# Variables
x = 10; name = "Anuj"

# Lists
nums = [1, 2, 3]; nums.append(4); nums[0]

# Tuples
point = (10, 20); x, y = point

# Dictionaries
d = {"a": 1, "b": 2}; d.get("c", 0)

# defaultdict
from collections import defaultdict
groups = defaultdict(list); groups["key"].append(1)

# Counter
from collections import Counter
freq = Counter(["a", "b", "a"]); freq.most_common(1)

# Sets
unique = {1, 2, 3}; 1 in unique

# List comprehensions
squares = [x**2 for x in range(10) if x % 2 == 0]

# zip
pairs = list(zip([1,2,3], ["a","b","c"]))

# Generators
gen = (x**2 for x in range(1000000))

# Lambda
double = lambda x: x * 2

# Functions with *args/**kwargs
def func(*args, **kwargs): pass

# Exception handling
try: risky_code()
except Exception as e: handle_error(e)

# Classes
class MyClass:
    def __init__(self, value): self.value = value
```

---

**End of Chapter 2**