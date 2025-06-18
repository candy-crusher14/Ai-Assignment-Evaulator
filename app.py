from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from collections import defaultdict
from sqlalchemy import func, and_, or_, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import google.generativeai as genai
import os
import hashlib
import re

warnings.filterwarnings("ignore")
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key')

# Set the template folder
app.template_folder = 'templates'

# SQLAlchemy Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///exam_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

db = SQLAlchemy(app)

# Configure Gemini API
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
# gemini_model = (genai.
                # GenerativeModel('gemini-pro'))


# Define database models
class Admin(db.Model):
    __tablename__ = 'admins'
    admin_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


class Student(db.Model):
    __tablename__ = 'students'
    student_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=func.current_timestamp())


class Teacher(db.Model):
    __tablename__ = 'teachers'
    teacher_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    department = db.Column(db.String(80))
    created_at = db.Column(db.DateTime, default=func.current_timestamp())


class Test(db.Model):
    __tablename__ = 'tests'
    test_id = db.Column(db.Integer, primary_key=True)
    test_name = db.Column(db.String(200), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.teacher_id'), nullable=False)
    created_at = db.Column(db.DateTime, default=func.current_timestamp())
    duration = db.Column(db.Integer)  # Duration in minutes
    instructions = db.Column(db.Text)


class Question(db.Model):
    __tablename__ = 'questions'
    question_id = db.Column(db.Integer, primary_key=True)
    question_text = db.Column(db.Text, nullable=False)
    test_id = db.Column(db.Integer, db.ForeignKey('tests.test_id'), nullable=False)
    max_score = db.Column(db.Integer, default=10)


class ExpectedAnswer(db.Model):
    __tablename__ = 'expected_answers'
    answer_id = db.Column(db.Integer, primary_key=True)
    answer_text = db.Column(db.Text, nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.question_id'), nullable=False)


class StudentAnswer(db.Model):
    __tablename__ = 'student_answers'
    answer_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.student_id'), nullable=False)
    test_id = db.Column(db.Integer, db.ForeignKey('tests.test_id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.question_id'), nullable=False)
    answer_text = db.Column(db.Text, nullable=False)
    score = db.Column(db.Float)
    evaluated_at = db.Column(db.DateTime)
    evaluation_method = db.Column(db.String(50))  # 'gemini' or 'algorithm'


class TeacherStudentRelationship(db.Model):
    __tablename__ = 'teacher_student_relationships'
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.teacher_id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('students.student_id'), nullable=False)


# Set English stopwords
EN_STOPWORDS = set(stopwords.words("english"))

# Preprocess text
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.naive_bayes import MultinomialNB

lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in EN_STOPWORDS]
    return tokens


def exact_match(expected_answer, student_answer):
    return int(expected_answer.strip().lower() == student_answer.strip().lower())


def partial_match(expected_answer, student_answer):
    expected_tokens = set(preprocess_text(expected_answer))
    student_tokens = set(preprocess_text(student_answer))
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & student_tokens) / len(expected_tokens)


def cosine_similarity_score(expected_answer, student_answer):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    try:
        tfidf_matrix = vectorizer.fit_transform([expected_answer, student_answer])
        cosine_sim = sk_cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return cosine_sim
    except:
        return 0.0


def sentiment_analysis(text):
    sentiment_score = sia.polarity_scores(text)['compound']
    return (sentiment_score + 1) / 2


def enhanced_sentence_match(expected_answer, student_answer):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings_expected = model.encode([expected_answer])
    embeddings_student = model.encode([student_answer])
    similarity = sk_cosine_similarity([embeddings_expected.flatten()], [embeddings_student.flatten()])[0][0]
    return similarity


def multinomial_naive_bayes_score(expected_answer, student_answer):
    try:
        vectorizer = CountVectorizer(tokenizer=preprocess_text)
        X = vectorizer.fit_transform([expected_answer, student_answer])
        y = [0, 1]
        clf = MultinomialNB()
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        return probs[1][1]
    except:
        return 0.5


def semantic_similarity_score(expected_answer, student_answer):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings_expected = model.encode([expected_answer])
    embeddings_student = model.encode([student_answer])
    similarity = sk_cosine_similarity([embeddings_expected.flatten()], [embeddings_student.flatten()])[0][0]
    return similarity


def coherence_score(expected_answer, student_answer):
    len_expected = len(word_tokenize(expected_answer))
    len_student = len(word_tokenize(student_answer))
    if max(len_expected, len_student) == 0:
        return 0.0
    return min(len_expected, len_student) / max(len_expected, len_student)


def relevance_score(expected_answer, student_answer):
    expected_tokens = set(word_tokenize(expected_answer.lower()))
    student_tokens = set(word_tokenize(student_answer.lower()))
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & student_tokens) / len(expected_tokens)


def evaluate_with_algorithm(expected, response, max_score=10):
    if expected.strip().lower() == response.strip().lower():
        return max_score, "algorithm"

    scores = [
        exact_match(expected, response) * max_score,
        partial_match(expected, response) * max_score,
        cosine_similarity_score(expected, response) * max_score,
        sentiment_analysis(response) * max_score,
        enhanced_sentence_match(expected, response) * max_score,
        multinomial_naive_bayes_score(expected, response) * max_score,
        semantic_similarity_score(expected, response) * max_score,
        coherence_score(expected, response) * max_score,
        relevance_score(expected, response) * max_score
    ]

    weights = [0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]

    final_score = sum(score * weight for score, weight in zip(scores, weights))
    return min(round(final_score, 2), max_score), "algorithm"


def evaluate_with_gemini(expected, response, question_text, max_score=10):
    prompt = f"""
    You are an expert evaluator for subjective exam answers. Evaluate the student's answer based on the question and expected answer.

    Question: {question_text}
    Expected Answer: {expected}
    Student's Answer: {response}

    Evaluation Criteria:
    1. Content accuracy and completeness (0-{max_score * 0.6} points)
    2. Relevance to question (0-{max_score * 0.2} points)
    3. Clarity and coherence (0-{max_score * 0.2} points)

    Provide the final score on a scale of 0 to {max_score}, and a brief evaluation comment.
    Format your response as: SCORE: [score]/10, COMMENT: [your comment]
    """

    try:
        response = gemini_model.generate_content(prompt)
        result = response.text
        match = re.search(r'SCORE:\s*(\d+\.?\d*)\s*/\s*10', result)
        if match:
            score = float(match.group(1))
            comment = result.split('COMMENT:', 1)[1].strip() if 'COMMENT:' in result else ""
            return min(score, max_score), "gemini", comment
    except Exception as e:
        print(f"Gemini evaluation error: {str(e)}")

    # Fallback to algorithm if Gemini fails
    score, _ = evaluate_with_algorithm(expected, response, max_score)
    return score, "algorithm", "Evaluation completed using backup method"


def evaluate_answer(student_answer_record):
    question = Question.query.get(student_answer_record.question_id)
    expected_answer = ExpectedAnswer.query.filter_by(
        question_id=student_answer_record.question_id
    ).first()

    if not expected_answer:
        return 0, "No expected answer found", ""

    max_score = question.max_score if question else 10

    # Check which evaluation method to use
    evaluation_method = request.form.get('evaluation_method', 'algorithm')

    if evaluation_method == 'gemini':
        score, method, comment = evaluate_with_gemini(
            expected_answer.answer_text,
            student_answer_record.answer_text,
            question.question_text if question else "",
            max_score
        )
        return score, method, comment

    score, method = evaluate_with_algorithm(
        expected_answer.answer_text,
        student_answer_record.answer_text,
        max_score
    )
    return score, method, ""


def check_test_taken(student_id, test_id):
    return StudentAnswer.query.filter_by(
        student_id=student_id,
        test_id=test_id
    ).first() is not None


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Homepage
@app.route('/')
def index():
    return render_template('homepage.html')


@app.route('/dashboard')
def dashboard():
    if 'admin_logged_in' in session:
        return redirect(url_for('admin_dashboard'))
    elif 'teacher_logged_in' in session:
        return redirect(url_for('teacher_dashboard'))
    elif 'student_logged_in' in session:
        return redirect(url_for('student_dashboard'))
    return redirect(url_for('index'))


# Admin routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        admin_code = request.form.get('admin_code', '')

        # Check admin code (replace 'ADMIN_SECRET' with your actual secret)
        if admin_code != os.environ.get('ADMIN_SECRET', 'admin123'):
            return render_template('admin_login.html', error='Invalid admin code')

        admin = Admin.query.filter_by(username=username, password=password).first()
        if admin:
            session['admin_logged_in'] = True
            session['admin_id'] = admin.admin_id
            return redirect(url_for('admin_dashboard'))
        return render_template('admin_login.html', error='Invalid credentials')
    return render_template('login.html')


@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    stats = {
        'students': Student.query.count(),
        'teachers': Teacher.query.count(),
        'tests': Test.query.count(),
        'recent_tests': Test.query.order_by(Test.created_at.desc()).limit(5).all()
    }
    return render_template('admin_dashboard.html', stats=stats)


@app.route('/admin/students', methods=['GET'])
def admin_students():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    students = Student.query.all()
    return render_template('admin_students.html', students=students)


@app.route('/admin/teachers', methods=['GET'])
def admin_teachers():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    teachers = Teacher.query.all()
    return render_template('admin_teachers.html', teachers=teachers)


@app.route('/admin/tests', methods=['GET'])
def admin_tests():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    tests = db.session.query(Test, Teacher).join(Teacher).all()
    return render_template('admin_tests.html', tests=tests)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_id', None)
    return redirect(url_for('index'))


# Teacher routes
@app.route('/teacher/register', methods=['GET', 'POST'])
def teacher_register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '')
        email = request.form.get('email', '')
        department = request.form.get('department', '')

        # Check password match
        if password != confirm_password:
            error = 'Passwords do not match'
        else:
            # Check if username exists
            if Student.query.filter_by(username=username).first() or Teacher.query.filter_by(username=username).first():
                error = 'Username already exists'
            else:
                # Hash password
                hashed_password = hash_password(password)

                # Create new teacher
                new_teacher = Teacher(
                    username=username,
                    password=hashed_password,
                    full_name=full_name,
                    email=email,
                    department=department
                )

                try:
                    db.session.add(new_teacher)
                    db.session.commit()
                    # Automatically log in after registration
                    session['teacher_logged_in'] = True
                    session['teacher_id'] = new_teacher.teacher_id
                    return redirect(url_for('teacher_dashboard'))
                except Exception as e:
                    db.session.rollback()
                    error = f'Registration failed: {str(e)}'

    return render_template('teacher_register.html', error=error)


@app.route('/teacher/login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        teacher = Teacher.query.filter_by(username=username, password=password).first()
        if teacher:
            session['teacher_logged_in'] = True
            session['teacher_id'] = teacher.teacher_id
            return redirect(url_for('teacher_dashboard'))
        return render_template('teacher_login.html', error='Invalid credentials')
    return render_template('teacher_login.html')


@app.route('/teacher/dashboard')
def teacher_dashboard():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    teacher = Teacher.query.get(teacher_id)

    stats = {
        'tests': Test.query.filter_by(teacher_id=teacher_id).count(),
        'students': TeacherStudentRelationship.query.filter_by(teacher_id=teacher_id).count(),
        'recent_tests': Test.query.filter_by(teacher_id=teacher_id)
        .order_by(Test.created_at.desc())
        .limit(5).all()
    }

    return render_template('teacher_dashboard.html', teacher=teacher, stats=stats)


@app.route('/teacher/tests', methods=['GET'])
def teacher_tests():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    tests = Test.query.filter_by(teacher_id=teacher_id).all()
    return render_template('teacher_tests.html', tests=tests)


@app.route('/teacher/test/create', methods=['GET', 'POST'])
def create_test():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']

    if request.method == 'POST':
        test_name = request.form['test_name']
        instructions = request.form.get('instructions', '')
        duration = int(request.form.get('duration', 60))

        new_test = Test(
            test_name=test_name,
            teacher_id=teacher_id,
            instructions=instructions,
            duration=duration
        )

        try:
            db.session.add(new_test)
            db.session.commit()
            flash('Test created successfully!', 'success')
            return redirect(url_for('edit_test', test_id=new_test.test_id))
        except Exception as e:
            db.session.rollback()
            return render_template('create_test.html', error=f'Failed to create test: {str(e)}')

    return render_template('create_test.html')


@app.route('/teacher/test/<int:test_id>/edit', methods=['GET', 'POST'])
def edit_test(test_id):
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    test = Test.query.filter_by(test_id=test_id, teacher_id=teacher_id).first()

    if not test:
        flash('Test not found or you do not have permission', 'error')
        return redirect(url_for('teacher_tests'))

    questions = Question.query.filter_by(test_id=test_id).all()
    question_answers = {}

    for question in questions:
        answers = ExpectedAnswer.query.filter_by(question_id=question.question_id).all()
        question_answers[question.question_id] = answers

    if request.method == 'POST':
        # Handle question updates
        for question in questions:
            question_text = request.form.get(f'question_{question.question_id}')
            if question_text:
                question.question_text = question_text

            # Update expected answers
            for answer in answers:
                answer_text = request.form.get(f'answer_{answer.answer_id}')
                if answer_text:
                    answer.answer_text = answer_text

        # Add new questions
        new_questions = request.form.getlist('new_question[]')
        new_question_answers = request.form.getlist('new_expected_answers[]')

        for i, question_text in enumerate(new_questions):
            if question_text.strip():
                new_question = Question(
                    question_text=question_text,
                    test_id=test_id,
                    max_score=int(request.form.getlist('new_max_score[]')[i])
                )
                db.session.add(new_question)
                db.session.flush()  # Get the new question ID

                answers = new_question_answers[i].split('|')
                for answer_text in answers:
                    if answer_text.strip():
                        new_answer = ExpectedAnswer(
                            answer_text=answer_text.strip(),
                            question_id=new_question.question_id
                        )
                        db.session.add(new_answer)

        try:
            db.session.commit()
            flash('Test updated successfully!', 'success')
            return redirect(url_for('edit_test', test_id=test_id))
        except Exception as e:
            db.session.rollback()
            return render_template('edit_test.html', test=test, questions=questions,
                                   question_answers=question_answers, error=f'Update failed: {str(e)}')

    return render_template('edit_test.html', test=test, questions=questions, question_answers=question_answers)


@app.route('/teacher/test/<int:test_id>/evaluate', methods=['GET'])
def evaluate_test(test_id):
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    test = Test.query.filter_by(test_id=test_id, teacher_id=teacher_id).first()

    if not test:
        flash('Test not found or you do not have permission', 'error')
        return redirect(url_for('teacher_tests'))

    # Get all student answers for this test
    student_answers = StudentAnswer.query.filter_by(test_id=test_id).all()

    # Evaluate each answer
    for answer in student_answers:
        if not answer.score:  # Only evaluate if not already scored
            score, method, _ = evaluate_answer(answer)
            answer.score = score
            answer.evaluation_method = method
            answer.evaluated_at = func.current_timestamp()

    try:
        db.session.commit()
        flash('Answers evaluated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Evaluation failed: {str(e)}', 'error')

    return redirect(url_for('view_test_results', test_id=test_id))


@app.route('/teacher/test/<int:test_id>/results', methods=['GET'])
def view_test_results(test_id):
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    test = Test.query.filter_by(test_id=test_id, teacher_id=teacher_id).first()

    if not test:
        flash('Test not found or you do not have permission', 'error')
        return redirect(url_for('teacher_tests'))

    # Get all student answers with student info
    results = db.session.query(
        StudentAnswer,
        Student,
        Question
    ).join(Student, StudentAnswer.student_id == Student.student_id
           ).join(Question, StudentAnswer.question_id == Question.question_id
                  ).filter(StudentAnswer.test_id == test_id).all()

    # Organize by student
    student_results = defaultdict(lambda: {'student': None, 'answers': [], 'total': 0})

    for answer, student, question in results:
        if not student_results[student.student_id]['student']:
            student_results[student.student_id]['student'] = student
        student_results[student.student_id]['answers'].append({
            'question': question,
            'answer': answer
        })
        student_results[student.student_id]['total'] += answer.score if answer.score else 0

    return render_template('test_results.html', test=test, student_results=student_results)


@app.route('/teacher/logout')
def teacher_logout():
    session.pop('teacher_logged_in', None)
    session.pop('teacher_id', None)
    return redirect(url_for('index'))


# Student routes
@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '')
        email = request.form.get('email', '')

        # Check password match
        if password != confirm_password:
            error = 'Passwords do not match'
        else:
            # Check if username exists
            if Student.query.filter_by(username=username).first() or Teacher.query.filter_by(username=username).first():
                error = 'Username already exists'
            else:
                # Hash password
                hashed_password = hash_password(password)

                # Create new student
                new_student = Student(
                    username=username,
                    password=hashed_password,
                    full_name=full_name,
                    email=email
                )

                try:
                    db.session.add(new_student)
                    db.session.commit()
                    # Automatically log in after registration
                    session['student_logged_in'] = True
                    session['student_id'] = new_student.student_id
                    return redirect(url_for('student_dashboard'))
                except Exception as e:
                    db.session.rollback()
                    error = f'Registration failed: {str(e)}'

    return render_template('student_register.html', error=error)


@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        student = Student.query.filter_by(username=username, password=password).first()
        if student:
            session['student_logged_in'] = True
            session['student_id'] = student.student_id
            return redirect(url_for('student_dashboard'))
        return render_template('student_login.html', error='Invalid credentials')
    return render_template('student_login.html')


@app.route('/student/dashboard')
def student_dashboard():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    student_id = session['student_id']
    student = Student.query.get(student_id)

    # Get assigned tests
    assigned_tests = db.session.query(Test).join(
        TeacherStudentRelationship,
        TeacherStudentRelationship.teacher_id == Test.teacher_id
    ).filter(
        TeacherStudentRelationship.student_id == student_id
    ).all()

    # Get recent test results
    recent_results = StudentAnswer.query.filter_by(
        student_id=student_id
    ).order_by(StudentAnswer.evaluated_at.desc()).limit(5).all()

    return render_template('student_dashboard.html', student=student,
                           assigned_tests=assigned_tests, recent_results=recent_results)


@app.route('/student/tests', methods=['GET'])
def student_tests():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    student_id = session['student_id']

    # Get tests assigned to this student
    assigned_tests = db.session.query(Test).join(
        TeacherStudentRelationship,
        TeacherStudentRelationship.teacher_id == Test.teacher_id
    ).filter(
        TeacherStudentRelationship.student_id == student_id
    ).all()

    # Check which tests have been taken
    for test in assigned_tests:
        test.taken = check_test_taken(student_id, test.test_id)

    return render_template('student_tests.html', tests=assigned_tests)


@app.route('/student/test/<int:test_id>/take', methods=['GET', 'POST'])
def take_test(test_id):
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    student_id = session['student_id']

    # Check if test is assigned to student
    relationship = TeacherStudentRelationship.query.filter_by(
        student_id=student_id,
        teacher_id=Test.query.get(test_id).teacher_id
    ).first()

    if not relationship:
        flash('You are not assigned to this test', 'error')
        return redirect(url_for('student_tests'))

    # Check if already taken
    if check_test_taken(student_id, test_id):
        flash('You have already taken this test', 'error')
        return redirect(url_for('student_tests'))

    test = Test.query.get(test_id)
    questions = Question.query.filter_by(test_id=test_id).all()

    if request.method == 'POST':
        for question in questions:
            answer_text = request.form.get(f'answer_{question.question_id}')
            if answer_text:
                new_answer = StudentAnswer(
                    student_id=student_id,
                    test_id=test_id,
                    question_id=question.question_id,
                    answer_text=answer_text
                )
                db.session.add(new_answer)

        try:
            db.session.commit()
            flash('Test submitted successfully!', 'success')
            return redirect(url_for('student_tests'))
        except Exception as e:
            db.session.rollback()
            return render_template('take_test.html', test=test, questions=questions,
                                   error=f'Submission failed: {str(e)}')

    return render_template('take_test.html', test=test, questions=questions)


@app.route('/student/results', methods=['GET'])
def student_results():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    student_id = session['student_id']

    # Get all test results
    results = db.session.query(
        StudentAnswer,
        Test,
        Question
    ).join(Test, StudentAnswer.test_id == Test.test_id
           ).join(Question, StudentAnswer.question_id == Question.question_id
                  ).filter(StudentAnswer.student_id == student_id).all()

    # Organize by test
    test_results = defaultdict(lambda: {'test': None, 'answers': [], 'total': 0})

    for answer, test, question in results:
        if not test_results[test.test_id]['test']:
            test_results[test.test_id]['test'] = test
        test_results[test.test_id]['answers'].append({
            'question': question,
            'answer': answer
        })
        test_results[test.test_id]['total'] += answer.score if answer.score else 0

    return render_template('student_results.html', test_results=test_results)


@app.route('/student/logout')
def student_logout():
    session.pop('student_logged_in', None)
    session.pop('student_id', None)
    return redirect(url_for('index'))


# Unified login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        role = request.form['role']
        admin_code = request.form.get('admin_code', '')

        if role == 'teacher':
            teacher = Teacher.query.filter_by(username=username, password=password).first()
            if teacher:
                session['teacher_logged_in'] = True
                session['teacher_id'] = teacher.teacher_id
                return redirect(url_for('teacher_dashboard'))
            error = 'Invalid credentials for teacher'

        elif role == 'student':
            student = Student.query.filter_by(username=username, password=password).first()
            if student:
                session['student_logged_in'] = True
                session['student_id'] = student.student_id
                return redirect(url_for('student_dashboard'))
            error = 'Invalid credentials for student'

        elif role == 'admin':
            # Check admin code (replace 'ADMIN_SECRET' with your actual secret)
            if admin_code != os.environ.get('ADMIN_SECRET', 'admin123'):
                error = 'Invalid admin code'
            else:
                admin = Admin.query.filter_by(username=username, password=password).first()
                if admin:
                    session['admin_logged_in'] = True
                    session['admin_id'] = admin.admin_id
                    return redirect(url_for('admin_dashboard'))
                error = 'Invalid credentials for admin'

    return render_template('login.html', error=error)


# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', message="Internal server error"), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        # Create default admin if not exists
        if not Admin.query.first():
            admin = Admin(
                username='admin',
                password=hash_password('admin123')
            )
            db.session.add(admin)
            db.session.commit()

    app.run(debug=True)