from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_sqlalchemy import SQLAlchemy
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
import warnings
from collections import defaultdict
from sqlalchemy import func, and_, or_, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import google.generativeai as genai
import os
import hashlib
import re
import random
import io
from datetime import datetime, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

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
#                 GenerativeModel('gemini-pro'))

# Define allowed departments and years
DEPARTMENTS = ('Computer Science', 'Information Technology')
YEARS = (1, 2, 3, 4)
SEMESTERS = ('Spring', 'Summer', 'Fall', 'Winter')


# Database Models
class Admin(db.Model):
    __tablename__ = 'admins'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


class Student(db.Model):
    __tablename__ = 'students'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    department = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=func.current_timestamp())
    submissions = db.relationship('Submission', backref='student', lazy=True)
    enrollments = db.relationship('Enrollment', backref='student', lazy=True)


class Teacher(db.Model):
    __tablename__ = 'teachers'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=func.current_timestamp())
    classrooms = db.relationship('Classroom', backref='teacher', lazy=True)
    assignments = db.relationship('Assignment', backref='teacher', lazy=True)
    overrides = db.relationship('Submission', foreign_keys='Submission.overridden_by', backref='overriding_teacher',
                                lazy=True)


class Classroom(db.Model):
    __tablename__ = 'classrooms'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    department = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=func.current_timestamp())
    assignments = db.relationship('Assignment', backref='classroom', lazy=True)
    enrollments = db.relationship('Enrollment', backref='classroom', lazy=True)

    @property
    def enrollments_count(self):
        return len(self.enrollments)

    @property
    def active_assignments(self):
        return [a for a in self.assignments if a.submission_deadline > datetime.now()]

    @property
    def completed_assignments(self):
        return [a for a in self.assignments if a.submission_deadline <= datetime.now()]

    @property
    def recent_assignments(self):
        return sorted(self.assignments, key=lambda a: a.created_at, reverse=True)[:3]


class Assignment(db.Model):
    __tablename__ = 'assignments'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    classroom_id = db.Column(db.Integer, db.ForeignKey('classrooms.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    instructions = db.Column(db.Text, nullable=False)
    submission_deadline = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=func.current_timestamp())
    semester = db.Column(db.String(50), nullable=False)
    evaluation_method = db.Column(db.String(50), nullable=False, default='gemini')
    expected_answer = db.Column(db.Text)
    max_score = db.Column(db.Float, default=10.0)
    submissions = db.relationship('Submission', backref='assignment', lazy=True)

    @property
    def is_completed(self):
        return self.submission_deadline < datetime.now()

    @property
    def submissions_submitted(self):
        return len([s for s in self.submissions if s.submitted_at is not None])

    @property
    def submissions_evaluated(self):
        return len([s for s in self.submissions if s.evaluated_at is not None])

    @property
    def submissions_pending_eval(self):
        return self.submissions_submitted - self.submissions_evaluated

    @property
    def submissions_not_submitted(self):
        return self.classroom.enrollments_count - self.submissions_submitted

    @property
    def submission_percentage(self):
        if self.classroom.enrollments_count == 0:
            return 0
        return (self.submissions_submitted / self.classroom.enrollments_count) * 100

    @property
    def evaluation_percentage(self):
        if self.submissions_submitted == 0:
            return 0
        return (self.submissions_evaluated / self.submissions_submitted) * 100

    @property
    def submission_status(self):
        if self.is_completed:
            return "Completed"
        return "Active"


class Submission(db.Model):
    __tablename__ = 'submissions'
    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignments.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    answer_text = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime)
    evaluated_at = db.Column(db.DateTime)
    score = db.Column(db.Float)
    evaluation_method = db.Column(db.String(50))
    gemini_comment = db.Column(db.Text)
    override_score = db.Column(db.Float)
    override_reason = db.Column(db.Text)
    overridden_at = db.Column(db.DateTime)
    overridden_by = db.Column(db.Integer, db.ForeignKey('teachers.id'))

    @property
    def display_score(self):
        return self.override_score if self.override_score is not None else self.score


class Enrollment(db.Model):
    __tablename__ = 'enrollments'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    classroom_id = db.Column(db.Integer, db.ForeignKey('classrooms.id'), nullable=False)
    __table_args__ = (db.UniqueConstraint('student_id', 'classroom_id', name='_stud_classroom_uc'),)


# Helper Functions
EN_STOPWORDS = set(stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()


def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
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
    len_expected = len(nltk.word_tokenize(expected_answer))
    len_student = len(nltk.word_tokenize(student_answer))
    if max(len_expected, len_student) == 0:
        return 0.0
    return min(len_expected, len_student) / max(len_expected, len_student)


def relevance_score(expected_answer, student_answer):
    expected_tokens = set(nltk.word_tokenize(expected_answer.lower()))
    student_tokens = set(nltk.word_tokenize(student_answer.lower()))
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & student_tokens) / len(expected_tokens)


def generate_expected_feedback(student_answer, expected_answer):
    similarity = semantic_similarity_score(expected_answer, student_answer)
    if similarity >= 0.8:
        return "Excellent match. All key points present."
    elif similarity >= 0.5:
        return "Fair answer. Some key points are missing."
    else:
        return "Low match. Several important concepts not found."


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

    score, _ = evaluate_with_algorithm(expected, response, max_score)
    return score, "algorithm", "Evaluation completed using backup method"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def check_assignment_taken(student_id, assignment_id):
    return Submission.query.filter_by(
        student_id=student_id,
        assignment_id=assignment_id
    ).first() is not None


# Routes
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


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        admin_code = request.form.get('admin_code', '')

        if admin_code != os.environ.get('ADMIN_SECRET', 'admin123'):
            return render_template('admin_login.html', error='Invalid admin code')

        admin = Admin.query.filter_by(username=username, password=password).first()
        if admin:
            session['admin_logged_in'] = True
            session['admin_id'] = admin.id
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
        'classrooms': Classroom.query.count(),
        'recent_assignments': Assignment.query.order_by(Assignment.created_at.desc()).limit(5).all()
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


@app.route('/admin/classrooms', methods=['GET'])
def admin_classrooms():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    classrooms = db.session.query(Classroom, Teacher).join(Teacher).all()
    return render_template('admin_classrooms.html', classrooms=classrooms)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_id', None)
    return redirect(url_for('index'))


@app.route('/teacher/register', methods=['GET', 'POST'])
def teacher_register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '')
        email = request.form.get('email', '')

        if password != confirm_password:
            error = 'Passwords do not match'
        else:
            if Student.query.filter_by(username=username).first() or Teacher.query.filter_by(username=username).first():
                error = 'Username already exists'
            else:
                hashed_password = hash_password(password)

                new_teacher = Teacher(
                    username=username,
                    password=hashed_password,
                    full_name=full_name,
                    email=email
                )

                try:
                    db.session.add(new_teacher)
                    db.session.commit()
                    session['teacher_logged_in'] = True
                    session['teacher_id'] = new_teacher.id
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
            session['teacher_id'] = teacher.id
            return redirect(url_for('teacher_dashboard'))
        return render_template('teacher_login.html', error='Invalid credentials')
    return render_template('teacher_login.html')


@app.route('/teacher/dashboard')
def teacher_dashboard():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    teacher = Teacher.query.get(teacher_id)

    # Get all classrooms for this teacher
    classrooms = Classroom.query.filter_by(teacher_id=teacher_id).all()

    # Get all assignments for this teacher
    assignments = Assignment.query.filter_by(teacher_id=teacher_id).all()

    # Get recent assignments (5 most recent)
    recent_assignments = Assignment.query.filter_by(teacher_id=teacher_id) \
        .order_by(Assignment.created_at.desc()).limit(5).all()

    # Calculate total enrollments
    enrollment_count = sum([c.enrollments_count for c in classrooms])

    # Get pending submissions
    pending_submissions = Submission.query.filter(
        Submission.score == None,
        Submission.assignment_id.in_([a.id for a in assignments])
    ).all()

    # Get completed submissions
    completed_submissions = Submission.query.filter(
        Submission.score != None,
        Submission.assignment_id.in_([a.id for a in assignments])
    ).all()

    # Get active assignments (deadline in future)
    active_assignments = [a for a in assignments if a.submission_deadline > datetime.now()]

    return render_template('teacher_dashboard.html',
                           teacher=teacher,
                           classrooms=classrooms,
                           assignments=assignments,
                           recent_assignments=recent_assignments,
                           total_enrollments=enrollment_count,
                           pending_evaluations=len(pending_submissions),
                           pending_submissions=pending_submissions,
                           completed_submissions=completed_submissions,
                           active_assignments=active_assignments)


@app.route('/teacher/classrooms', methods=['GET'])
def teacher_classrooms():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    classrooms = Classroom.query.filter_by(teacher_id=teacher_id).all()
    return render_template('teacher_classrooms.html', classrooms=classrooms)


@app.route('/teacher/classroom/create', methods=['GET', 'POST'])
def create_classroom():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']

    if request.method == 'POST':
        name = request.form['name']
        department = request.form['department']
        year = int(request.form['year'])

        new_classroom = Classroom(
            name=name,
            teacher_id=teacher_id,
            department=department,
            year=year
        )

        try:
            db.session.add(new_classroom)
            db.session.commit()
            flash('Classroom created successfully!', 'success')
            return redirect(url_for('teacher_classrooms'))
        except Exception as e:
            db.session.rollback()
            return render_template('create_classroom.html', error=f'Failed to create classroom: {str(e)}')

    return render_template('create_classroom.html', departments=DEPARTMENTS, years=YEARS)


@app.route('/teacher/assignment/create', methods=['GET', 'POST'])
def create_assignment():
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    teacher = Teacher.query.get(teacher_id)
    classrooms = Classroom.query.filter_by(teacher_id=teacher_id).all()

    if request.method == 'POST':
        title = request.form['title']
        classroom_id = int(request.form['classroom_id'])
        instructions = request.form['instructions']
        deadline_days = int(request.form['deadline_days'])
        semester = request.form['semester']
        evaluation_method = request.form['evaluation_method']
        expected_answer = request.form.get('expected_answer', '')
        max_score = float(request.form.get('max_score', 10.0))

        # Calculate deadline
        submission_deadline = datetime.now() + timedelta(days=deadline_days)

        new_assignment = Assignment(
            title=title,
            classroom_id=classroom_id,
            teacher_id=teacher_id,
            instructions=instructions,
            submission_deadline=submission_deadline,
            semester=semester,
            evaluation_method=evaluation_method,
            expected_answer=expected_answer,
            max_score=max_score
        )

        try:
            db.session.add(new_assignment)
            db.session.commit()
            flash('Assignment created successfully!', 'success')
            return redirect(url_for('teacher_dashboard'))
        except Exception as e:
            db.session.rollback()
            return render_template('create_assignment.html',
                                   classrooms=classrooms,
                                   semesters=SEMESTERS,
                                   error=f'Failed to create assignment: {str(e)}')

    return render_template('create_assignment.html',
                           classrooms=classrooms,
                           semesters=SEMESTERS)


@app.route('/teacher/assignment/<int:assignment_id>/evaluate', methods=['GET'])
def evaluate_assignment(assignment_id):
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    assignment = Assignment.query.filter_by(id=assignment_id, teacher_id=teacher_id).first()

    if not assignment:
        flash('Assignment not found or you do not have permission', 'error')
        return redirect(url_for('teacher_dashboard'))

    submissions = Submission.query.filter_by(assignment_id=assignment_id).all()

    for submission in submissions:
        if not submission.score:
            if assignment.evaluation_method == 'gemini':
                score, method, comment = evaluate_with_gemini(
                    assignment.expected_answer,
                    submission.answer_text,
                    assignment.instructions,
                    assignment.max_score
                )
                submission.score = score
                submission.evaluation_method = method
                submission.gemini_comment = comment
            else:
                score, method = evaluate_with_algorithm(
                    assignment.expected_answer,
                    submission.answer_text,
                    assignment.max_score
                )
                submission.score = score
                submission.evaluation_method = method
            submission.evaluated_at = func.current_timestamp()

    try:
        db.session.commit()
        flash('Submissions evaluated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Evaluation failed: {str(e)}', 'error')

    return redirect(url_for('view_assignment_results', assignment_id=assignment_id))


@app.route('/teacher/assignment/<int:assignment_id>/results', methods=['GET'])
def view_assignment_results(assignment_id):
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    assignment = Assignment.query.filter_by(id=assignment_id, teacher_id=teacher_id).first()

    if not assignment:
        flash('Assignment not found or you do not have permission', 'error')
        return redirect(url_for('teacher_dashboard'))

    results = db.session.query(
        Submission,
        Student
    ).join(Student, Submission.student_id == Student.id
           ).filter(Submission.assignment_id == assignment_id).all()

    student_results = defaultdict(lambda: {'student': None, 'submissions': []})

    for submission, student in results:
        if not student_results[student.id]['student']:
            student_results[student.id]['student'] = student

        display_score = submission.display_score
        student_results[student.id]['submissions'].append({
            'submission': submission,
            'display_score': display_score
        })

    return render_template('assignment_results.html', assignment=assignment, student_results=student_results)


@app.route('/teacher/override/<int:submission_id>', methods=['GET', 'POST'])
def override_score(submission_id):
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.get(submission.assignment_id)
    student = Student.query.get(submission.student_id)

    if request.method == 'POST':
        try:
            new_score = float(request.form['new_score'])
            reason = request.form['reason']

            if new_score < 0 or new_score > assignment.max_score:
                flash(f'Score must be between 0 and {assignment.max_score}', 'error')
                return redirect(url_for('override_score', submission_id=submission_id))

            submission.override_score = new_score
            submission.override_reason = reason
            submission.overridden_at = func.now()
            submission.overridden_by = session['teacher_id']

            db.session.commit()
            flash('Score override successful!', 'success')
            return redirect(url_for('view_assignment_results', assignment_id=submission.assignment_id))
        except Exception as e:
            db.session.rollback()
            flash(f'Override failed: {str(e)}', 'error')

    return render_template('override_form.html', submission=submission, assignment=assignment, student=student)


@app.route('/teacher/report/<int:assignment_id>')
def teacher_report(assignment_id):
    if not session.get('teacher_logged_in'):
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    assignment = Assignment.query.filter_by(id=assignment_id, teacher_id=teacher_id).first_or_404()
    teacher = Teacher.query.get(teacher_id)

    submissions = db.session.query(
        Submission,
        Student
    ).join(Student, Submission.student_id == Student.id
           ).filter(Submission.assignment_id == assignment_id).all()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,
        spaceAfter=12
    )

    header_style = ParagraphStyle(
        'Header',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=6
    )

    story = []

    story.append(Paragraph(f"Assignment Evaluation Report", title_style))
    story.append(Paragraph(f"Assignment: {assignment.title}", header_style))
    story.append(Paragraph(f"Teacher: {teacher.full_name}", body_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 0.2 * inch))

    table_data = [['Student', 'Score', 'Method', 'Override', 'Final', 'Date']]

    for submission, student in submissions:
        display_score = submission.display_score
        table_data.append([
            student.full_name,
            f"{submission.score:.1f}" if submission.score is not None else "N/A",
            "Gemini" if submission.evaluation_method == 'gemini' else "Algorithm",
            f"{submission.override_score:.1f}" if submission.override_score is not None else "-",
            f"{display_score:.1f}",
            submission.evaluated_at.strftime('%Y-%m-%d') if submission.evaluated_at else "-"
        ])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))

    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    total_submissions = len(submissions)
    override_count = sum(1 for s in submissions if s[0].override_score is not None)
    avg_score = sum(s[0].display_score for s in submissions) / total_submissions if total_submissions else 0

    story.append(Paragraph(f"<b>Summary Statistics:</b>", header_style))
    story.append(Paragraph(f"Total Submissions: {total_submissions}", body_style))
    story.append(Paragraph(f"Overridden Scores: {override_count}", body_style))
    story.append(Paragraph(f"Average Score: {avg_score:.2f}", body_style))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name=f"assignment_{assignment_id}_report.pdf")


@app.route('/teacher/logout')
def teacher_logout():
    session.pop('teacher_logged_in', None)
    session.pop('teacher_id', None)
    return redirect(url_for('index'))


@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '')
        email = request.form.get('email', '')
        department = request.form['department']
        year = int(request.form['year'])

        if department not in DEPARTMENTS:
            error = 'Invalid department'
        elif year not in YEARS:
            error = 'Invalid year'
        elif password != confirm_password:
            error = 'Passwords do not match'
        else:
            if Student.query.filter_by(username=username).first() or Teacher.query.filter_by(username=username).first():
                error = 'Username already exists'
            else:
                hashed_password = hash_password(password)

                new_student = Student(
                    username=username,
                    password=hashed_password,
                    full_name=full_name,
                    email=email,
                    department=department,
                    year=year
                )

                try:
                    db.session.add(new_student)
                    db.session.commit()
                    session['student_logged_in'] = True
                    session['student_id'] = new_student.id
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
            session['student_id'] = student.id
            return redirect(url_for('student_dashboard'))
        return render_template('student_login.html', error='Invalid credentials')
    return render_template('student_login.html')


@app.route('/student/dashboard')
def student_dashboard():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    student_id = session['student_id']
    student = Student.query.get(student_id)

    # Get classrooms student is enrolled in
    enrolled_classrooms = Classroom.query.join(Enrollment).filter(
        Enrollment.student_id == student_id
    ).all()

    # Get assignments from enrolled classrooms
    assignments = Assignment.query.filter(
        Assignment.classroom_id.in_([c.id for c in enrolled_classrooms])
    ).all()

    for assignment in assignments:
        assignment.taken = check_assignment_taken(student_id, assignment.id)

    recent_results = Submission.query.filter_by(
        student_id=student_id
    ).order_by(Submission.evaluated_at.desc()).limit(5).all()

    for result in recent_results:
        result.display_score = result.display_score

    return render_template('student_dashboard.html', student=student,
                           assignments=assignments, recent_results=recent_results)


@app.route('/student/assignments', methods=['GET'])
def student_assignments():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    student_id = session['student_id']
    student = Student.query.get(student_id)

    enrolled_classrooms = Classroom.query.join(Enrollment).filter(
        Enrollment.student_id == student_id
    ).all()

    assignments = Assignment.query.filter(
        Assignment.classroom_id.in_([c.id for c in enrolled_classrooms])
    ).all()

    for assignment in assignments:
        assignment.taken = check_assignment_taken(student_id, assignment.id)

    return render_template('student_assignments.html', assignments=assignments)


@app.route('/student/assignment/<int:assignment_id>/submit', methods=['GET', 'POST'])
def submit_assignment(assignment_id):
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    student_id = session['student_id']
    student = Student.query.get(student_id)
    assignment = Assignment.query.get(assignment_id)

    # Check if student is enrolled in the classroom
    is_enrolled = Enrollment.query.filter_by(
        student_id=student_id,
        classroom_id=assignment.classroom_id
    ).first() is not None

    if not is_enrolled:
        flash('You are not enrolled in this classroom', 'error')
        return redirect(url_for('student_assignments'))

    if check_assignment_taken(student_id, assignment_id):
        flash('You have already submitted this assignment', 'error')
        return redirect(url_for('student_assignments'))

    if request.method == 'POST':
        answer_text = request.form.get('answer_text')
        if answer_text:
            new_submission = Submission(
                student_id=student_id,
                assignment_id=assignment_id,
                answer_text=answer_text,
                submitted_at=func.current_timestamp()
            )
            db.session.add(new_submission)

        try:
            db.session.commit()
            flash('Assignment submitted successfully!', 'success')
            return redirect(url_for('student_assignments'))
        except Exception as e:
            db.session.rollback()
            return render_template('submit_assignment.html', assignment=assignment,
                                   error=f'Submission failed: {str(e)}')

    return render_template('submit_assignment.html', assignment=assignment)


@app.route('/student/report')
def student_report():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    student_id = session['student_id']
    student = Student.query.get(student_id)

    submissions = db.session.query(
        Submission,
        Assignment
    ).join(Assignment, Submission.assignment_id == Assignment.id
           ).filter(Submission.student_id == student_id).all()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,
        spaceAfter=12
    )

    header_style = ParagraphStyle(
        'Header',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=6
    )

    story = []

    story.append(Paragraph(f"Student Evaluation Report", title_style))
    story.append(Paragraph(f"Student: {student.full_name}", header_style))
    story.append(Paragraph(f"Department: {student.department}, Year: {student.year}", body_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 0.2 * inch))

    total_score = 0
    max_possible = 0

    for submission, assignment in submissions:
        display_score = submission.display_score

        story.append(Paragraph(f"<b>Assignment:</b> {assignment.title}", header_style))
        story.append(Paragraph(f"<b>Instructions:</b> {assignment.instructions}", body_style))
        story.append(Paragraph(f"<b>Your Answer:</b> {submission.answer_text}", body_style))

        story.append(Paragraph(f"<b>Score:</b> {display_score:.1f}/{assignment.max_score}", body_style))
        story.append(Paragraph(
            f"<b>Evaluation Method:</b> {'Gemini API' if submission.evaluation_method == 'gemini' else 'Algorithm'}",
            body_style))

        if submission.evaluation_method == 'gemini' and submission.gemini_comment:
            remarks = submission.gemini_comment
        else:
            remarks = generate_expected_feedback(submission.answer_text, assignment.expected_answer)

        story.append(Paragraph(f"<b>Remarks:</b> {remarks}", body_style))
        story.append(
            Paragraph(f"<b>Evaluated At:</b> {submission.evaluated_at.strftime('%Y-%m-%d %H:%M')}", body_style))

        if submission.override_score is not None:
            teacher = Teacher.query.get(submission.overridden_by)
            teacher_name = teacher.full_name if teacher else "Unknown"
            story.append(Paragraph(
                f"<b>Score Override:</b> {submission.override_score} (by {teacher_name} on {submission.overridden_at.strftime('%Y-%m-%d')})",
                body_style))
            story.append(Paragraph(f"<b>Reason:</b> {submission.override_reason}", body_style))

        story.append(Spacer(1, 0.1 * inch))

        total_score += display_score
        max_possible += assignment.max_score

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"<b>Overall Performance:</b>", header_style))
    story.append(Paragraph(f"Total Score: {total_score:.1f}/{max_possible}", body_style))
    story.append(
        Paragraph(f"Average Score: {total_score / len(submissions) if submissions else 0:.1f} per assignment",
                  body_style))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="student_report.pdf")


@app.route('/student/results', methods=['GET'])
def student_results():
    if not session.get('student_logged_in'):
        return redirect(url_for('student_login'))

    student_id = session['student_id']

    results = db.session.query(
        Submission,
        Assignment
    ).join(Assignment, Submission.assignment_id == Assignment.id
           ).filter(Submission.student_id == student_id).all()

    assignment_results = defaultdict(lambda: {'assignment': None, 'submission': None})

    for submission, assignment in results:
        if not assignment_results[assignment.id]['assignment']:
            assignment_results[assignment.id]['assignment'] = assignment
        assignment_results[assignment.id]['submission'] = submission

    return render_template('student_results.html', assignment_results=assignment_results)


@app.route('/student/logout')
def student_logout():
    session.pop('student_logged_in', None)
    session.pop('student_id', None)
    return redirect(url_for('index'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        admin_code = request.form.get('admin_code', '')

        hashed_password = hash_password(password)

        if role == 'teacher':
            teacher = Teacher.query.filter_by(username=username, password=hashed_password).first()
            if teacher:
                session['teacher_logged_in'] = True
                session['teacher_id'] = teacher.id
                return redirect(url_for('teacher_dashboard'))
            error = 'Invalid credentials for teacher'

        elif role == 'student':
            student = Student.query.filter_by(username=username, password=hashed_password).first()
            if student:
                session['student_logged_in'] = True
                session['student_id'] = student.id
                return redirect(url_for('student_dashboard'))
            error = 'Invalid credentials for student'

        elif role == 'admin':
            if admin_code != os.environ.get('ADMIN_SECRET', 'admin123'):
                error = 'Invalid admin code'
            else:
                admin = Admin.query.filter_by(username=username, password=hashed_password).first()
                if admin:
                    session['admin_logged_in'] = True
                    session['admin_id'] = admin.id
                    return redirect(url_for('admin_dashboard'))
                error = 'Invalid credentials for admin'

    return render_template('login.html', error=error)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', message="Internal server error"), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        if not Admin.query.first():
            admin = Admin(
                username='admin',
                password=hash_password('admin123')
            )
            db.session.add(admin)
            db.session.commit()

        # Create a test teacher if none exists
        if not Teacher.query.first():
            teacher = Teacher(
                username='teacher',
                password=hash_password('teacher123'),
                full_name='Test Teacher'
            )
            db.session.add(teacher)
            db.session.commit()

        # Create a test classroom if none exists
        if not Classroom.query.first() and Teacher.query.first():
            classroom = Classroom(
                name='Computer Science 101',
                teacher_id=1,
                department='Computer Science',
                year=1
            )
            db.session.add(classroom)
            db.session.commit()

    app.run(debug=True)