from pandas import DataFrame, concat


subjects = {
    "Physics": [
        "What is the speed of light?",
        "Explain Newton's laws.",
        "What is quantum mechanics?",
        "How does gravity work?",
        "What are the types of energy?",
        "Describe the theory of relativity.",
        "What is thermodynamics?",
        "What is electromagnetism?",
        "Explain wave-particle duality.",
        "What is the difference between mass and weight?",
    ],
    "Chemistry": [
        "What is the formula for water?",
        "What is an atom?",
        "Explain chemical bonding.",
        "What are acids and bases?",
        "What is the periodic table?",
        "What are the states of matter?",
        "Describe a chemical reaction.",
        "What is pH?",
        "What is organic chemistry?",
        "What is a mole in chemistry?",
    ],
    "Geography": [
        "What is the capital of France?",
        "Describe the layers of the Earth.",
        "What are the continents?",
        "What is the significance of the equator?",
        "Where is the Great Barrier Reef?",
        "What is the highest mountain in the world?",
        "What are renewable energy sources?",
        "What is a tectonic plate?",
        "Explain climate change.",
        "What is urbanization?",
    ],
    "History": [
        "Who discovered America?",
        "What caused World War II?",
        "When was the Declaration of Independence signed?",
        "Who was Julius Caesar?",
        "What was the Renaissance?",
        "Who were the ancient Egyptians?",
        "What is the significance of the Berlin Wall?",
        "Describe the Cold War.",
        "Who was Martin Luther King Jr.?",
        "What is feudalism?",
    ],
    "Biology": [
        "What is photosynthesis?",
        "Explain cell structure.",
        "What is DNA?",
        "Describe the process of evolution.",
        "What are the main functions of proteins?",
        "What is a virus?",
        "What is the human genome?",
        "What are ecosystems?",
        "Explain natural selection.",
        "What are the stages of mitosis?",
    ],
    "Astronomy": [
        "What is a black hole?",
        "What is the solar system?",
        "Explain the life cycle of a star.",
        "What is the Big Bang Theory?",
        "What are exoplanets?",
        "What is dark matter?",
        "What causes a solar eclipse?",
        "What is a comet?",
        "Describe the Milky Way galaxy.",
        "What is the Hubble Space Telescope?",
    ],
    "Art": [
        "Who painted the Mona Lisa?",
        "What is Impressionism?",
        "Describe the elements of design.",
        "What is modern art?",
        "Who was Vincent van Gogh?",
        "What is sculpture?",
        "Explain the concept of perspective.",
        "What are the different styles of painting?",
        "Who is Frida Kahlo?",
        "What is abstract art?",
    ],
}

df = DataFrame()
for sub in subjects:
    temp = DataFrame({'sentence': subjects[sub], 'subject': [sub]*len(subjects[sub])})
    df = concat([df, temp])

df.to_csv('./data/sub_clas.csv')