# few_shot_examples.py

# A list of 20 in-context True/False examples for P(True) prompting.
FEW_SHOT_EXAMPLES = [
    {"statement": "The Earth revolves around the Sun.",      "label": "True"},
    {"statement": "The Moon is made of cheese.",             "label": "False"},
    {"statement": "Water freezes at 0°C.",                   "label": "True"},
    {"statement": "Humans can breathe underwater without equipment.", "label": "False"},
    {"statement": "Light travels faster than sound.",         "label": "True"},
    {"statement": "Bats are mammals.",                        "label": "True"},
    {"statement": "Gold is heavier than lead.",               "label": "False"},
    {"statement": "Venus is the closest planet to the Sun.",  "label": "False"},
    {"statement": "Mount Everest is the tallest mountain on Earth.", "label": "True"},
    {"statement": "The Great Wall of China is visible from space without aid.", "label": "False"},
    {"statement": "Pi is exactly 3.14.",                      "label": "False"},
    {"statement": "The human body has four lungs.",           "label": "False"},
    {"statement": "A group of crows is called a murder.",     "label": "True"},
    {"statement": "Venus is the hottest planet in the solar system.", "label": "True"},
    {"statement": "Shakespeare wrote 37 plays in his lifetime.", "label": "True"},
    {"statement": "Lightning never strikes the same place twice.", "label": "False"},
    {"statement": "Adult humans have 206 bones.",             "label": "True"},
    {"statement": "The Atlantic Ocean is the largest ocean on Earth.", "label": "False"},
    {"statement": "Albert Einstein won the Nobel Prize in Physics in 1921.", "label": "True"},
    {"statement": "Sound travels faster in air than in water.", "label": "False"}
]
