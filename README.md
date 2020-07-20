# Resume-NER

# About

This repository applies BERT for named entity recognition on resumes. The goal is to find useful information present in resume.

# Requirements

```bash
pip3 install -r requirements.txt
```

# Training

To train model use:
```bash
python3 train.py
``` 
optional arguments:

-e epochs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of epochs

-o path &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; output path to save model state

# Flask REST API

```bash
python3 server/app.py -p path-for-model-weights
```
API will be live at localhost:5000 endpoint predict.

### cURL request

```bash
curl --location --request POST 'http://localhost:5000/predict' --form 'resume=@/resume-path.pdf'
```

Output:
```json
{
  "entities": [
    {
      "entity": "Name",
      "start": 3,
      "end": 19,
      "text": "Ayush Srivastava"
    },
    {
      "entity": "Designation",
      "start": 22,
      "end": 35,
      "text": "Web Developer"
    },
    {
      "entity": "Degree",
      "start": 50,
      "end": 56,
      "text": "B.Tech"
    },
    {
      "entity": "Years of Experience",
      "start": 72,
      "end": 73,
      "text": "3"
    },
    {
      "entity": "Designation",
      "start": 582,
      "end": 590,
      "text": "Software"
    },
    {
      "entity": "College Name",
      "start": 937,
      "end": 943,
      "text": "JSSATE"
    },
    {
      "entity": "Degree",
      "start": 955,
      "end": 961,
      "text": "B.Tech"
    },
    {
      "entity": "Graduation Year",
      "start": 964,
      "end": 968,
      "text": "2016"
    },
    {
      "entity": "Skills",
      "start": 1040,
      "end": 1050,
      "text": "Algorithms"
    },
    {
      "entity": "Skills",
      "start": 1054,
      "end": 1069,
      "text": "Data Structures"
    },
    {
      "entity": "Skills",
      "start": 1074,
      "end": 1088,
      "text": "SQL based DBMS"
    },
    {
      "entity": "Skills",
      "start": 1094,
      "end": 1109,
      "text": "Turing Machines"
    },
    {
      "entity": "Skills",
      "start": 1114,
      "end": 1125,
      "text": "OS Concepts"
    },
    {
      "entity": "Skills",
      "start": 1168,
      "end": 1185,
      "text": "C/C++ Programming"
    },
    {
      "entity": "Skills",
      "start": 1188,
      "end": 1219,
      "text": "○ Designing (UI/UX & Photoshop)"
    },
    {
      "entity": "Skills",
      "start": 1221,
      "end": 1251,
      "text": "○ Web Development (HTML & CSS)"
    },
    {
      "entity": "Skills",
      "start": 1266,
      "end": 1272,
      "text": "SKILLS"
    },
    {
      "entity": "Skills",
      "start": 1287,
      "end": 1323,
      "text": "ReactJS, Gatsby, jQuery, JavaScript,"
    },
    {
      "entity": "Skills",
      "start": 1326,
      "end": 1360,
      "text": "HTML, CSS, Materialize, Bootstrap,"
    },
    {
      "entity": "Skills",
      "start": 1363,
      "end": 1371,
      "text": "Backend:"
    },
    {
      "entity": "Skills",
      "start": 1373,
      "end": 1378,
      "text": "Flask"
    },
    {
      "entity": "Skills",
      "start": 1381,
      "end": 1400,
      "text": "Data Visualization:"
    },
    {
      "entity": "Skills",
      "start": 1402,
      "end": 1416,
      "text": "D3, Matplotlib"
    },
    {
      "entity": "Skills",
      "start": 1419,
      "end": 1431,
      "text": "Cloud Suite:"
    },
    {
      "entity": "Skills",
      "start": 1433,
      "end": 1454,
      "text": "Google Cloud Platform"
    },
    {
      "entity": "Skills",
      "start": 1457,
      "end": 1479,
      "text": "Programming Languages:"
    },
    {
      "entity": "Skills",
      "start": 1481,
      "end": 1488,
      "text": "C++, C,"
    },
    {
      "entity": "Skills",
      "start": 1491,
      "end": 1497,
      "text": "Python"
    },
    {
      "entity": "Skills",
      "start": 1511,
      "end": 1521,
      "text": "Oracle SQL"
    },
    {
      "entity": "Skills",
      "start": 1524,
      "end": 1557,
      "text": "Data Structures and Algorithms in"
    },
    {
      "entity": "Skills",
      "start": 1560,
      "end": 1563,
      "text": "C++"
    }
  ]
}
```
