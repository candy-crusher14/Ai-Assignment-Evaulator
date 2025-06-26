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
from sqlalchemy import func, and_, or_, text, event
from sqlalchemy.exc import OperationalError, SQLAlchemyError, IntegrityError, DataError
import google.generativeai as genai
import os
import hashlib
import re
import random
import io
import uuid  # For invite codes
from datetime import datetime, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_for_dev_123!@#')

# Removed the problematic session clearing block that was here

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///exam_system.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 300}

db = SQLAlchemy(app)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None
    print("Warning: GEMINI_API_KEY not found. Gemini evaluation will not be available.")

DEPARTMENTS = ('Computer Science', 'Information Technology')
YEARS = (1, 2, 3, 4)
SEMESTERS_LIST = [f"Semester {i}" for i in range(1, 9)]


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
    email = db.Column(db.String(120), unique=True)
    department = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    submissions = db.relationship('Submission', backref='student', lazy='dynamic')
    enrollments = db.relationship('Enrollment', backref='student', lazy='dynamic', cascade="all, delete-orphan")


class Teacher(db.Model):
    __tablename__ = 'teachers'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    classrooms = db.relationship('Classroom', backref='teacher', lazy='dynamic', cascade="all, delete-orphan")
    assignments = db.relationship('Assignment', backref='teacher', lazy='dynamic', cascade="all, delete-orphan")
    overrides = db.relationship('Submission', foreign_keys='Submission.overridden_by', backref='overriding_teacher',
                                lazy='dynamic')


def generate_invite_code():
    return str(uuid.uuid4().hex)[:8].upper()


class Classroom(db.Model):
    __tablename__ = 'classrooms'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    department = db.Column(db.String(80), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=True)
    invite_code = db.Column(db.String(8), unique=True, nullable=False, default=generate_invite_code)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    assignments = db.relationship('Assignment', backref='classroom', lazy='dynamic', cascade="all, delete-orphan")
    enrollments = db.relationship('Enrollment', backref='classroom', lazy='dynamic', cascade="all, delete-orphan")

    @property
    def enrollments_count(self): return self.enrollments.count()

    @property
    def active_assignments(self): return self.assignments.filter(
        Assignment.submission_deadline > datetime.utcnow()).all()

    @property
    def completed_assignments(self): return self.assignments.filter(
        Assignment.submission_deadline <= datetime.utcnow()).all()

    @property
    def recent_assignments(self): return self.assignments.order_by(Assignment.created_at.desc()).limit(3).all()


@event.listens_for(Classroom, 'before_insert')
def ensure_unique_invite_code(mapper, connection, target):
    if not target.invite_code or Classroom.query.filter_by(invite_code=target.invite_code).first() is not None:
        while True:
            new_code = generate_invite_code()
            if Classroom.query.filter_by(invite_code=new_code).first() is None:
                target.invite_code = new_code
                break


class Assignment(db.Model):
    __tablename__ = 'assignments'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    classroom_id = db.Column(db.Integer, db.ForeignKey('classrooms.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    instructions = db.Column(db.Text, nullable=False)
    submission_deadline = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    semester = db.Column(db.String(50), nullable=False)
    evaluation_method = db.Column(db.String(50), nullable=False, default='gemini')
    expected_answer = db.Column(db.Text, nullable=True)
    max_score = db.Column(db.Float, default=10.0)
    submissions = db.relationship('Submission', backref='assignment', lazy='dynamic', cascade="all, delete-orphan")

    @property
    def is_completed(self): return self.submission_deadline < datetime.utcnow()

    @property
    def submissions_submitted_count(self): return self.submissions.filter(Submission.submitted_at != None).count()

    @property
    def submissions_evaluated_count(self): return self.submissions.filter(Submission.evaluated_at != None).count()

    @property
    def submissions_pending_eval_count(self): return max(0,
                                                         self.submissions_submitted_count - self.submissions_evaluated_count)

    @property
    def submissions_total_possible(self): return self.classroom.enrollments_count if self.classroom else 0

    @property
    def submissions_not_submitted_count(self): return max(0,
                                                          self.submissions_total_possible - self.submissions_submitted_count)

    @property
    def submission_percentage(self): return (
                                                        self.submissions_submitted_count / self.submissions_total_possible) * 100 if self.submissions_total_possible > 0 else 0

    @property
    def evaluation_percentage(self): return (
                                                        self.submissions_evaluated_count / self.submissions_submitted_count) * 100 if self.submissions_submitted_count > 0 else 0

    @property
    def submission_status_text(self): return "Completed" if self.is_completed else "Active"


class Submission(db.Model):
    __tablename__ = 'submissions'
    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignments.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    answer_text = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime, nullable=True)
    evaluated_at = db.Column(db.DateTime, nullable=True)
    score = db.Column(db.Float, nullable=True)
    evaluation_method = db.Column(db.String(50), nullable=True)
    gemini_comment = db.Column(db.Text, nullable=True)
    override_score = db.Column(db.Float, nullable=True)
    override_reason = db.Column(db.Text, nullable=True)
    overridden_at = db.Column(db.DateTime, nullable=True)
    overridden_by = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=True)

    @property
    def display_score(self): return self.override_score if self.override_score is not None else self.score


class Enrollment(db.Model):
    __tablename__ = 'enrollments'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    classroom_id = db.Column(db.Integer, db.ForeignKey('classrooms.id'), nullable=False)
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('student_id', 'classroom_id', name='_enrollment_stud_classroom_uc'),)


# Helper Functions
EN_STOPWORDS = set(stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()


def preprocess_text(text):
    if not text: return []
    tokens = nltk.word_tokenize(str(text))
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in EN_STOPWORDS]
    return tokens


def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()


def evaluate_with_algorithm(expected, response_text, max_score=10.0):
    if not expected or not response_text:
        return 0.0, "algorithm", "Missing expected answer or student response."
    expected_str, response_str = str(expected), str(response_text)
    if expected_str.strip().lower() == response_str.strip().lower():
        return float(max_score), "algorithm", "Exact match."
    expected_tokens = set(preprocess_text(expected_str))
    student_tokens = set(preprocess_text(response_str))
    if not expected_tokens:
        partial_score_val = 0.0
    else:
        partial_score_val = len(expected_tokens & student_tokens) / len(expected_tokens)
    final_score = partial_score_val * float(max_score)
    feedback = generate_expected_feedback(response_str, expected_str)  # This uses semantic similarity
    return min(round(final_score, 2), float(max_score)), "algorithm", feedback


def semantic_similarity_score(expected_answer, student_answer):
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        if not expected_answer or not student_answer: return 0.0
        embeddings_expected = model.encode([str(expected_answer)])
        embeddings_student = model.encode([str(student_answer)])
        similarity = sk_cosine_similarity(embeddings_expected, embeddings_student)[0][0]
        return similarity
    except Exception as e:
        print(f"Error in semantic_similarity_score: {e}", file=sys.stderr)
        return 0.0


def generate_expected_feedback(student_answer, expected_answer):
    similarity = semantic_similarity_score(expected_answer, student_answer)
    if similarity >= 0.8:
        return "Excellent match. All key points present."
    elif similarity >= 0.5:
        return "Fair answer. Some key points are missing."
    else:
        return "Low match. Several important concepts not found."


def evaluate_with_gemini(expected_ans, student_response, question_instructions, max_score=10.0):
    if not gemini_model:
        score, method, comment = evaluate_with_algorithm(expected_ans, student_response, max_score)
        return score, "algorithm_fallback", f"Gemini API not configured. {comment}"
    prompt = f"""
    You are an expert evaluator for subjective exam answers. Evaluate the student's answer based on the question instructions and, if provided, an expected answer.
    Question/Instructions: {question_instructions}
    Expected Answer (if available, use as a guideline): {expected_ans if expected_ans else "Not provided. Evaluate based on question/instructions."}
    Student's Answer: {student_response}
    Evaluation Criteria:
    1. Content accuracy and completeness relative to the question and expected answer (if provided).
    2. Relevance of the answer to the question.
    3. Clarity, coherence, and structure of the answer.
    Provide the final score on a scale of 0 to {max_score}.
    Also, provide a concise evaluation comment explaining the score.
    Format your response strictly as: SCORE: [score]/{max_score}, COMMENT: [your comment]
    Example: SCORE: 7.5/{max_score}, COMMENT: The student understood the main concepts but missed a few details.
    """
    try:
        api_response = gemini_model.generate_content(prompt)
        result_text = api_response.text
        score_match = re.search(r'SCORE:\s*(\d+\.?\d*)\s*/\s*\d+\.?\d*', result_text)
        comment_match = re.search(r'COMMENT:\s*(.*)', result_text, re.IGNORECASE)
        if score_match and comment_match:
            score = float(score_match.group(1))
            comment = comment_match.group(1).strip()
            return min(score, float(max_score)), "gemini", comment
        else:
            print(f"Gemini response parsing failed. Response: {result_text}", file=sys.stderr)
            score, _, alg_comment = evaluate_with_algorithm(expected_ans, student_response, max_score)
            return score, "algorithm_fallback", f"Gemini parsing error. {alg_comment}"
    except Exception as e:
        print(f"Gemini API evaluation error: {str(e)}", file=sys.stderr)
        score, _, alg_comment = evaluate_with_algorithm(expected_ans, student_response, max_score)
        return score, "algorithm_fallback", f"Gemini API error. {alg_comment}"


# Routes
@app.route('/')
def index():
    if 'admin_logged_in' in session: return redirect(url_for('admin_dashboard'))
    if 'teacher_logged_in' in session: return redirect(url_for('teacher_dashboard'))
    if 'student_logged_in' in session: return redirect(url_for('student_dashboard'))
    return render_template('homepage.html')


@app.route('/dashboard')
def dashboard():
    if 'admin_logged_in' in session:
        return redirect(url_for('admin_dashboard'))
    elif 'teacher_logged_in' in session:
        return redirect(url_for('teacher_dashboard'))
    elif 'student_logged_in' in session:
        return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))


# Admin Routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        admin_code = request.form.get('admin_code', '')
        if admin_code != os.environ.get('ADMIN_SECRET', 'admin123'):
            flash('Invalid admin code', 'danger')
            return render_template('login.html', error='Invalid admin code', default_role='admin')
        admin = Admin.query.filter_by(username=username, password=password).first()
        if admin:
            session['admin_logged_in'], session['admin_id'] = True, admin.id
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('Invalid admin credentials', 'danger')
    return render_template('login.html', default_role='admin')


@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        flash('Please log in as admin.', 'warning');
        return redirect(url_for('login', role='admin'))
    stats = {
        'students': Student.query.count(), 'teachers': Teacher.query.count(),
        'classrooms': Classroom.query.count(), 'assignments': Assignment.query.count(),
        'recent_assignments': Assignment.query.order_by(Assignment.created_at.desc()).limit(5).all()
    }
    return render_template('admin_dashboard.html', stats=stats)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None);
    session.pop('admin_id', None)
    flash('Admin logged out successfully.', 'info');
    return redirect(url_for('index'))


# Teacher Routes
@app.route('/teacher/register', methods=['GET', 'POST'])
def teacher_register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        error = None
        if not all([username, password, confirm_password, full_name, email]):
            error = 'All fields are required.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif Teacher.query.filter(func.lower(Teacher.username) == func.lower(username)).first() or \
                Student.query.filter(func.lower(Student.username) == func.lower(username)).first():
            error = 'Username already exists.'
        elif Teacher.query.filter(func.lower(Teacher.email) == func.lower(email)).first() or \
                Student.query.filter(func.lower(Student.email) == func.lower(email)).first():
            error = 'Email already registered.'
        else:
            new_teacher = Teacher(username=username, password=hash_password(password), full_name=full_name, email=email)
            try:
                db.session.add(new_teacher);
                db.session.commit()
                session['teacher_logged_in'], session['teacher_id'] = True, new_teacher.id
                flash('Teacher registration successful! Welcome.', 'success');
                return redirect(url_for('teacher_dashboard'))
            except IntegrityError as ie:
                db.session.rollback();
                error = f'Registration failed: Username or email might already exist.';
                app.logger.error(f"IntegrityError: {ie}")
            except Exception as e:
                db.session.rollback(); error = f'Registration error: {e}'; app.logger.error(f"Exception: {e}")
        if error: flash(error, 'danger')
    return render_template('teacher_register.html')


@app.route('/teacher/login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        teacher = Teacher.query.filter_by(username=username, password=password).first()
        if teacher:
            session['teacher_logged_in'], session['teacher_id'] = True, teacher.id
            flash('Teacher login successful!', 'success');
            return redirect(url_for('teacher_dashboard'))
        flash('Invalid teacher credentials.', 'danger')
    return render_template('login.html', default_role='teacher')


@app.route('/teacher/dashboard')
def teacher_dashboard():
    if not session.get('teacher_logged_in'):
        flash('Please log in as a teacher.', 'warning');
        return redirect(url_for('login', role='teacher'))
    teacher_id = session['teacher_id']
    teacher = Teacher.query.get_or_404(teacher_id)

    filter_classroom_id = request.args.get('filter_classroom_id', type=int)
    filter_assignment_id = request.args.get('filter_assignment_id', type=int)
    active_tab_on_load = request.args.get('active_tab', 'teacher-dashboard')

    classrooms = teacher.classrooms.order_by(Classroom.name).all()
    all_assignments_for_filter_dropdown = teacher.assignments.order_by(Assignment.title).all()

    assignments_base_query = teacher.assignments
    if filter_classroom_id:
        assignments_base_query = assignments_base_query.filter(Assignment.classroom_id == filter_classroom_id)

    report_assignments = assignments_base_query.order_by(Assignment.created_at.desc()).all()
    if filter_assignment_id:
        report_assignments = [a for a in report_assignments if a.id == filter_assignment_id]

    active_assignments_list_query = teacher.assignments.filter(Assignment.submission_deadline > datetime.utcnow())
    if filter_classroom_id and active_tab_on_load == 'teacher-dashboard':
        active_assignments_list_query = active_assignments_list_query.filter(
            Assignment.classroom_id == filter_classroom_id)
    active_assignments_list = active_assignments_list_query.order_by(Assignment.submission_deadline.asc()).all()

    total_enrollments = sum(c.enrollments_count for c in classrooms)
    assignment_ids_by_teacher = [a.id for a in teacher.assignments.all()]

    pending_submissions_query = Submission.query.filter(
        Submission.assignment_id.in_(assignment_ids_by_teacher),
        Submission.submitted_at != None, Submission.evaluated_at == None
    )
    completed_submissions_query = Submission.query.filter(
        Submission.assignment_id.in_(assignment_ids_by_teacher),
        Submission.evaluated_at != None
    )

    if filter_classroom_id:
        assignments_in_filtered_classroom_ids = [a.id for a in
                                                 Assignment.query.filter_by(classroom_id=filter_classroom_id,
                                                                            teacher_id=teacher_id).all()]
        pending_submissions_query = pending_submissions_query.filter(
            Submission.assignment_id.in_(assignments_in_filtered_classroom_ids))
        completed_submissions_query = completed_submissions_query.filter(
            Submission.assignment_id.in_(assignments_in_filtered_classroom_ids))

    if filter_assignment_id:
        pending_submissions_query = pending_submissions_query.filter(Submission.assignment_id == filter_assignment_id)
        completed_submissions_query = completed_submissions_query.filter(
            Submission.assignment_id == filter_assignment_id)

    pending_submissions = pending_submissions_query.order_by(Submission.submitted_at.desc()).all()
    completed_submissions = completed_submissions_query.order_by(Submission.evaluated_at.desc()).all()

    return render_template('teacher_dashboard.html',
                           teacher=teacher, classrooms=classrooms,
                           assignments=report_assignments,
                           all_assignments_for_filter=all_assignments_for_filter_dropdown,
                           active_assignments=active_assignments_list,
                           total_enrollments=total_enrollments,
                           pending_evaluations=len(pending_submissions),
                           pending_submissions=pending_submissions,
                           completed_submissions=completed_submissions,
                           current_filter_classroom_id=filter_classroom_id,
                           current_filter_assignment_id=filter_assignment_id,
                           active_tab_on_load=active_tab_on_load
                           )


@app.route('/teacher/classroom/create', methods=['POST'])
def create_classroom():
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger');
        return redirect(url_for('login', role='teacher'))
    teacher_id = session['teacher_id']
    name = request.form.get('name', '').strip()
    department = request.form.get('department')
    year_str = request.form.get('year')
    description = request.form.get('description', '').strip()
    if not name or not department or not year_str:
        flash('Classroom name, department, and year are required.', 'danger')
    else:
        try:
            year = int(year_str)
            if year not in YEARS:
                flash('Invalid year selected.', 'danger')
            elif department not in DEPARTMENTS:
                flash('Invalid department selected.', 'danger')
            else:
                new_classroom = Classroom(name=name, teacher_id=teacher_id, department=department, year=year,
                                          description=description)
                try:
                    db.session.add(new_classroom);
                    db.session.commit()
                    flash(f'Classroom "{name}" created successfully! Invite Code: {new_classroom.invite_code}',
                          'success')
                except IntegrityError:
                    db.session.rollback();
                    flash(f'Classroom name or invite code might conflict. Try a different name.', 'warning')
                except Exception as e:
                    db.session.rollback(); flash(f'Failed to create classroom: {e}', 'danger')
        except ValueError:
            flash('Invalid year format.', 'danger')
    return redirect(url_for('teacher_dashboard', active_tab='classroom-manager'))


@app.route('/teacher/assignment/create', methods=['POST'])
def create_assignment():
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger');
        return redirect(url_for('login', role='teacher'))
    teacher_id = session['teacher_id']
    title = request.form.get('title', '').strip()
    classroom_id_str = request.form.get('classroom_id')
    instructions = request.form.get('instructions', '').strip()
    deadline_days_str = request.form.get('deadline_days')
    semester = request.form.get('semester')
    evaluation_method = request.form.get('evaluation_method')
    expected_answer = request.form.get('expected_answer', '').strip()
    max_score_str = request.form.get('max_score')

    if not all([title, classroom_id_str, instructions, deadline_days_str, semester, evaluation_method, max_score_str]):
        flash('All assignment fields are required.', 'danger')
    else:
        try:
            classroom_id = int(classroom_id_str)
            deadline_days = int(deadline_days_str)
            max_score = float(max_score_str)
            if deadline_days <= 0:
                flash('Deadline must be at least 1 day.', 'danger')
            elif max_score <= 0:
                flash('Max score must be positive.', 'danger')
            else:
                classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=teacher_id).first()
                if not classroom:
                    flash('Selected classroom not found or not yours.', 'danger')
                elif evaluation_method == 'expected' and not expected_answer:
                    flash('Expected answer is required for "Expected Answer Matching" method.', 'danger')
                else:
                    submission_deadline = datetime.utcnow() + timedelta(days=deadline_days)
                    new_assignment = Assignment(
                        title=title, classroom_id=classroom_id, teacher_id=teacher_id, instructions=instructions,
                        submission_deadline=submission_deadline, semester=semester, evaluation_method=evaluation_method,
                        expected_answer=expected_answer if evaluation_method == 'expected' else None,
                        max_score=max_score
                    )
                    try:
                        db.session.add(new_assignment);
                        db.session.commit()
                        flash(f'Assignment "{title}" created successfully!', 'success')
                    except Exception as e:
                        db.session.rollback();
                        flash(f'Failed to create assignment: {str(e)}', 'danger')
        except ValueError:
            flash('Invalid number format for classroom ID, deadline days, or max score.', 'danger')
    return redirect(url_for('teacher_dashboard', active_tab='create-assignment'))


@app.route('/teacher/assignment/<int:assignment_id>/evaluate_all', methods=['POST'])
def evaluate_all_submissions_for_assignment(assignment_id):
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger');
        return redirect(url_for('login', role='teacher'))
    teacher_id = session['teacher_id']
    assignment = Assignment.query.filter_by(id=assignment_id, teacher_id=teacher_id).first_or_404()
    submissions_to_evaluate = assignment.submissions.filter(Submission.submitted_at != None,
                                                            Submission.evaluated_at == None).all()
    if not submissions_to_evaluate:
        flash('No pending submissions to evaluate for this assignment.', 'info')
    else:
        evaluated_count = 0
        for sub in submissions_to_evaluate:
            eval_details = evaluate_with_gemini(assignment.expected_answer, sub.answer_text, assignment.instructions,
                                                assignment.max_score) \
                if assignment.evaluation_method == 'gemini' and gemini_model \
                else evaluate_with_algorithm(assignment.expected_answer, sub.answer_text, assignment.max_score)
            sub.score, sub.evaluation_method, sub.gemini_comment = eval_details[0], eval_details[1], eval_details[
                2] if len(eval_details) > 2 else None
            if sub.evaluation_method == 'algorithm' and len(
                eval_details) > 2 and not sub.gemini_comment: sub.gemini_comment = eval_details[2]
            sub.evaluated_at = datetime.utcnow();
            evaluated_count += 1
        try:
            db.session.commit();
            flash(f'{evaluated_count} submissions for "{assignment.title}" evaluated successfully!', 'success')
        except Exception as e:
            db.session.rollback(); flash(f'Evaluation process encountered an error: {str(e)}', 'danger')
    return redirect(url_for('teacher_dashboard', filter_assignment_id=assignment_id, active_tab='evaluation-hub'))


@app.route('/teacher/submission/<int:submission_id>/evaluate_single', methods=['POST'])
def evaluate_single_submission(submission_id):
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger');
        return redirect(url_for('login', role='teacher'))
    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.get_or_404(submission.assignment_id)
    if assignment.teacher_id != session['teacher_id']:
        flash('You do not have permission to evaluate this submission.', 'danger')
        return redirect(url_for('teacher_dashboard', active_tab='evaluation-hub'))
    if submission.evaluated_at:
        flash('This submission has already been evaluated.', 'info')
    else:
        eval_details = evaluate_with_gemini(assignment.expected_answer, submission.answer_text, assignment.instructions,
                                            assignment.max_score) \
            if assignment.evaluation_method == 'gemini' and gemini_model \
            else evaluate_with_algorithm(assignment.expected_answer, submission.answer_text, assignment.max_score)
        submission.score, submission.evaluation_method, submission.gemini_comment = eval_details[0], eval_details[1], \
        eval_details[2] if len(eval_details) > 2 else None
        if submission.evaluation_method == 'algorithm' and len(
            eval_details) > 2 and not submission.gemini_comment: submission.gemini_comment = eval_details[2]
        submission.evaluated_at = datetime.utcnow()
        try:
            db.session.commit()
            flash(f'Submission by {submission.student.full_name} for "{assignment.title}" evaluated successfully!',
                  'success')
        except Exception as e:
            db.session.rollback();
            flash(f'Error evaluating submission: {str(e)}', 'danger')
    return redirect(
        url_for('teacher_dashboard', filter_assignment_id=assignment.id, filter_classroom_id=assignment.classroom_id,
                active_tab='evaluation-hub'))


@app.route('/teacher/override/<int:submission_id>', methods=['POST'])
def override_score(submission_id):
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger');
        return redirect(url_for('login', role='teacher'))
    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.get_or_404(submission.assignment_id)
    if assignment.teacher_id != session['teacher_id']:
        flash('You do not have permission to override this score.', 'danger')
    else:
        try:
            new_score_str = request.form.get('new_score')
            reason = request.form.get('reason', '').strip()
            if not new_score_str or not reason:
                flash('New score and reason are required.', 'danger')
            else:
                new_score = float(new_score_str)
                if not (0 <= new_score <= assignment.max_score):
                    flash(f'Score must be 0-{assignment.max_score}.', 'danger')
                else:
                    submission.override_score = new_score;
                    submission.override_reason = reason
                    submission.overridden_at = datetime.utcnow();
                    submission.overridden_by = session['teacher_id']
                    submission.evaluated_at = datetime.utcnow()
                    db.session.commit();
                    flash('Score override successful!', 'success')
        except ValueError:
            flash('Invalid score format.', 'danger')
        except Exception as e:
            db.session.rollback(); flash(f'Override failed: {e}', 'danger')
    return redirect(
        url_for('teacher_dashboard', filter_assignment_id=assignment.id, filter_classroom_id=assignment.classroom_id,
                active_tab='evaluation-hub'))


@app.route('/teacher/report/assignment/<int:assignment_id>')
def teacher_report_assignment(assignment_id):
    if not session.get('teacher_logged_in'): return redirect(url_for('login', role='teacher'))
    teacher_id = session['teacher_id']
    assignment = Assignment.query.filter_by(id=assignment_id, teacher_id=teacher_id).first_or_404()
    teacher = Teacher.query.get(teacher_id)
    submissions_with_students = db.session.query(Submission, Student) \
        .join(Student, Submission.student_id == Student.id) \
        .filter(Submission.assignment_id == assignment_id) \
        .order_by(Student.full_name).all()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles, story = getSampleStyleSheet(), []
    title_style = ParagraphStyle('Title', parent=styles['h1'], alignment=1, spaceAfter=12, fontSize=18)
    header_style = ParagraphStyle('Header', parent=styles['h2'], fontSize=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=6, leading=12)
    small_body_style = ParagraphStyle('SmallBody', parent=styles['Normal'], fontSize=8, spaceAfter=4, leading=10)
    story.extend([
        Paragraph(f"Assignment Evaluation Report", title_style),
        Paragraph(f"<b>Assignment:</b> {assignment.title}", header_style),
        Paragraph(f"<b>Classroom:</b> {assignment.classroom.name}", body_style),
        Paragraph(f"<b>Teacher:</b> {teacher.full_name}", body_style),
        Paragraph(f"<b>Max Score:</b> {assignment.max_score}", body_style),
        Paragraph(f"<b>Deadline:</b> {assignment.submission_deadline.strftime('%Y-%m-%d %H:%M')} UTC", body_style),
        Paragraph(f"<b>Generated on:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", body_style),
        Spacer(1, 0.2 * inch)
    ])
    table_data = [['Student', 'Submitted', 'Score', 'Method', 'Override', 'Final', 'Evaluated']]
    for sub, stud in submissions_with_students:
        table_data.append([
            Paragraph(stud.full_name, small_body_style),
            Paragraph(sub.submitted_at.strftime('%y-%m-%d %H:%M') if sub.submitted_at else "No", small_body_style),
            Paragraph(f"{sub.score:.1f}" if sub.score is not None else "N/A", small_body_style),
            Paragraph(sub.evaluation_method if sub.evaluation_method else "N/A", small_body_style),
            Paragraph(f"{sub.override_score:.1f}" if sub.override_score is not None else "-", small_body_style),
            Paragraph(f"{sub.display_score:.1f}" if sub.display_score is not None else "N/A", small_body_style),
            Paragraph(sub.evaluated_at.strftime('%y-%m-%d %H:%M') if sub.evaluated_at else "-", small_body_style)
        ])
    if not submissions_with_students:
        story.append(Paragraph("No submissions found.", body_style))
    else:
        table = Table(table_data,
                      colWidths=[1.5 * inch, 1 * inch, 0.7 * inch, 1 * inch, 0.7 * inch, 0.7 * inch, 1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8), ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#DCE6F1")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black), ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]));
        story.append(table)
    story.append(Spacer(1, 0.2 * inch))
    valid_scores = [s.display_score for s, _ in submissions_with_students if s.display_score is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    story.extend([
        Paragraph(f"<b>Summary Statistics:</b>", header_style),
        Paragraph(f"Total Possible Submissions (Enrolled): {assignment.submissions_total_possible}", body_style),
        Paragraph(f"Actual Submissions: {assignment.submissions_submitted_count}", body_style),
        Paragraph(f"Submissions Evaluated: {assignment.submissions_evaluated_count}", body_style),
        Paragraph(f"Scores Overridden: {assignment.submissions.filter(Submission.override_score != None).count()}",
                  body_style),
        Paragraph(f"Average Final Score (of evaluated): {avg_score:.2f}", body_style)
    ])
    doc.build(story);
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"assignment_{assignment.id}_report.pdf",
                     mimetype='application/pdf')


@app.route('/teacher/logout')
def teacher_logout():
    session.pop('teacher_logged_in', None);
    session.pop('teacher_id', None)
    flash('Teacher logged out successfully.', 'info');
    return redirect(url_for('index'))


@app.route('/teacher/classroom/<int:classroom_id>/manage', methods=['GET'])
def manage_classroom_details(classroom_id):
    if not session.get('teacher_logged_in'):
        flash('Please log in.', 'warning');
        return redirect(url_for('login', role='teacher'))
    flash(f"Classroom management (ID: {classroom_id}) feature is currently under development.", "info")
    return redirect(url_for('teacher_dashboard', active_tab='classroom-manager'))


@app.route('/teacher/classroom/<int:classroom_id>/remove_student/<int:student_id>', methods=['POST'])
def remove_student_from_classroom(classroom_id, student_id):
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger');
        return redirect(url_for('login', role='teacher'))
    flash("Student removal feature is part of classroom management, which is under development.", "info")
    return redirect(url_for('teacher_dashboard', active_tab='classroom-manager'))


# Student Routes
@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        department = request.form.get('department')
        year_str = request.form.get('year')
        error = None
        if not all([username, password, confirm_password, full_name, email, department, year_str]):
            error = 'All fields are required.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif Student.query.filter(func.lower(Student.username) == func.lower(username)).first() or \
                Teacher.query.filter(func.lower(Teacher.username) == func.lower(username)).first():
            error = 'Username already exists.'
        elif Student.query.filter(func.lower(Student.email) == func.lower(email)).first() or \
                Teacher.query.filter(func.lower(Teacher.email) == func.lower(email)).first():
            error = 'Email already registered.'
        else:
            try:
                year = int(year_str)
                if department not in DEPARTMENTS:
                    error = 'Invalid department.'
                elif year not in YEARS:
                    error = 'Invalid year.'
                else:
                    new_student = Student(username=username, password=hash_password(password), full_name=full_name,
                                          email=email, department=department, year=year)
                    db.session.add(new_student);
                    db.session.commit()
                    session['student_logged_in'], session['student_id'] = True, new_student.id
                    flash('Student registration successful! Welcome.', 'success');
                    return redirect(url_for('student_dashboard'))
            except ValueError:
                error = 'Invalid year format.'
            except IntegrityError as ie:
                db.session.rollback(); error = f'Registration failed: Username or email might already exist.'; app.logger.error(
                    f"IntegrityError: {ie}")
            except Exception as e:
                db.session.rollback(); error = f'Registration error: {e}'; app.logger.error(f"Exception: {e}")
        if error: flash(error, 'danger')
    return render_template('student_register.html', departments=DEPARTMENTS, years=YEARS)


@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        student = Student.query.filter_by(username=username, password=password).first()
        if student:
            session['student_logged_in'], session['student_id'] = True, student.id
            flash('Student login successful!', 'success');
            return redirect(url_for('student_dashboard'))
        flash('Invalid student credentials.', 'danger')
    return render_template('login.html', default_role='student')


@app.route('/student/dashboard')
def student_dashboard():
    if not session.get('student_logged_in'):
        flash('Please log in as a student.', 'warning');
        return redirect(url_for('login', role='student'))
    student = Student.query.get_or_404(session['student_id'])
    enrolled_classrooms_ids = [e.classroom_id for e in student.enrollments.all()]
    enrolled_classrooms = Classroom.query.filter(Classroom.id.in_(enrolled_classrooms_ids)).all()
    submitted_as_ids = db.session.query(Submission.assignment_id).filter(Submission.student_id == student.id,
                                                                         Submission.submitted_at != None)
    available_assignments = Assignment.query.filter(
        Assignment.classroom_id.in_(enrolled_classrooms_ids),
        Assignment.submission_deadline > datetime.utcnow(),
        ~Assignment.id.in_(submitted_as_ids)
    ).order_by(Assignment.submission_deadline.asc()).all()
    recent_submissions = Submission.query.filter_by(student_id=student.id) \
        .filter(Submission.submitted_at != None) \
        .join(Assignment).order_by(Submission.submitted_at.desc()).limit(5).all()
    return render_template('student_dashboard.html', student=student, available_assignments=available_assignments,
                           recent_submissions=recent_submissions, enrolled_classrooms=enrolled_classrooms)


@app.route('/student/join_classroom', methods=['GET', 'POST'])
def student_join_classroom():
    if not session.get('student_logged_in'):
        flash('Please log in to join a classroom.', 'warning');
        return redirect(url_for('login', role='student'))
    student_id = session['student_id']
    student = Student.query.get_or_404(student_id)
    if request.method == 'POST':
        invite_code = request.form.get('invite_code', '').strip().upper()
        if not invite_code:
            flash('Invite code is required.', 'danger')
        else:
            classroom = Classroom.query.filter_by(invite_code=invite_code).first()
            if not classroom:
                flash('Invalid invite code. Classroom not found.', 'danger')
            elif classroom.year != student.year or classroom.department != student.department:
                flash(
                    f'You cannot join this classroom. It is for Year {classroom.year} {classroom.department} students.',
                    'danger')
            else:
                existing_enrollment = Enrollment.query.filter_by(student_id=student_id,
                                                                 classroom_id=classroom.id).first()
                if existing_enrollment:
                    flash(f'You are already enrolled in "{classroom.name}".', 'info')
                else:
                    try:
                        enrollment = Enrollment(student_id=student_id, classroom_id=classroom.id)
                        db.session.add(enrollment);
                        db.session.commit()
                        flash(f'Successfully joined classroom "{classroom.name}"!', 'success');
                        return redirect(url_for('student_dashboard'))
                    except IntegrityError:
                        db.session.rollback(); flash('Could not join: already enrolled.', 'warning')
                    except Exception as e:
                        db.session.rollback(); flash(f'Error joining: {e}', 'danger')
    return render_template('student_join_classroom.html', student=student)


@app.route('/student/assignment/<int:assignment_id>/submit', methods=['GET', 'POST'])
def submit_assignment(assignment_id):
    if not session.get('student_logged_in'):
        flash('Please log in to submit.', 'warning');
        return redirect(url_for('login', role='student'))
    student_id = session['student_id']
    assignment = Assignment.query.get_or_404(assignment_id)
    student = Student.query.get_or_404(student_id)
    if not Enrollment.query.filter_by(student_id=student_id, classroom_id=assignment.classroom_id).first():
        flash('You are not enrolled for this assignment.', 'danger');
        return redirect(url_for('student_dashboard'))
    existing_submission = Submission.query.filter_by(student_id=student_id, assignment_id=assignment_id).first()
    if existing_submission and existing_submission.submitted_at:
        flash('You have already submitted this assignment.', 'info');
        return redirect(url_for('student_assignments_list'))
    if assignment.submission_deadline < datetime.utcnow():
        flash('The deadline for this assignment has passed.', 'danger');
        return redirect(url_for('student_assignments_list'))
    if request.method == 'POST':
        answer_text = request.form.get('answer_text', '').strip()
        if not answer_text:
            flash('Answer text cannot be empty.', 'danger')
        else:
            target_submission = existing_submission or Submission(assignment_id=assignment_id, student_id=student_id)
            target_submission.answer_text = answer_text;
            target_submission.submitted_at = datetime.utcnow()
            if not existing_submission: db.session.add(target_submission)
            try:
                db.session.commit();
                flash('Assignment submitted successfully!', 'success');
                return redirect(url_for('student_assignments_list'))
            except Exception as e:
                db.session.rollback(); flash(f'Error submitting assignment: {str(e)}', 'danger')
    return render_template('submit_assignment.html', assignment=assignment, student=student)


@app.route('/student/report/pdf')
def student_report_pdf():
    if not session.get('student_logged_in'): return redirect(url_for('login', role='student'))
    student = Student.query.get_or_404(session['student_id'])
    submissions_with_assignments = db.session.query(Submission, Assignment) \
        .join(Assignment).filter(Submission.student_id == student.id, Submission.submitted_at != None) \
        .order_by(Assignment.submission_deadline.desc()).all()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles, story = getSampleStyleSheet(), []
    title_style = ParagraphStyle('Title', parent=styles['h1'], alignment=1, spaceAfter=12, fontSize=18)
    header_style = ParagraphStyle('Header', parent=styles['h2'], fontSize=12, spaceAfter=6, leading=14)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=6, leading=12)
    answer_style = ParagraphStyle('Answer', parent=styles['Normal'], fontSize=9, spaceAfter=4, leading=11,
                                  leftIndent=10, backColor=colors.HexColor("#f0f0f0"), padding=5,
                                  borderColor=colors.lightgrey, borderWidth=1)
    story.extend([
        Paragraph(f"Student Performance Report", title_style),
        Paragraph(f"<b>Student:</b> {student.full_name}", header_style),
        Paragraph(f"<b>ID:</b> {student.username} | <b>Email:</b> {student.email}", body_style),
        Paragraph(f"<b>Department:</b> {student.department} | <b>Year:</b> {student.year}", body_style),
        Paragraph(f"<b>Report Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", body_style),
        Spacer(1, 0.2 * inch)
    ])
    total_score_achieved, total_max_score_possible, evaluated_assignments_count = 0, 0, 0
    if not submissions_with_assignments:
        story.append(Paragraph("No submissions found.", body_style))
    else:
        for sub, assign in submissions_with_assignments:
            story.extend([
                Paragraph(f"<u>Assignment: {assign.title}</u> (Class: {assign.classroom.name})", header_style),
                Paragraph(
                    f"<b>Submitted:</b> {sub.submitted_at.strftime('%Y-%m-%d %H:%M') if sub.submitted_at else 'N/A'}",
                    body_style),
                Paragraph("<b>Your Answer:</b>", body_style),
                Paragraph(sub.answer_text if sub.answer_text else "<i>No answer text.</i>", answer_style)
            ])
            if sub.evaluated_at:
                story.extend([
                    Paragraph(f"<b>Evaluated:</b> {sub.evaluated_at.strftime('%Y-%m-%d %H:%M')}", body_style),
                    Paragraph(
                        f"<b>Score:</b> {sub.display_score if sub.display_score is not None else 'N/A'} / {assign.max_score}",
                        body_style),
                    Paragraph(f"<b>Method:</b> {sub.evaluation_method or 'N/A'}", body_style)
                ])
                if sub.gemini_comment: story.append(Paragraph(f"<b>Feedback:</b> {sub.gemini_comment}", body_style))
                if sub.override_score is not None:
                    teacher_name = Teacher.query.get(
                        sub.overridden_by).full_name if sub.overridden_by and Teacher.query.get(
                        sub.overridden_by) else "Teacher"
                    story.extend([
                        Paragraph(
                            f"<b>Overridden Score:</b> {sub.override_score} by {teacher_name} on {sub.overridden_at.strftime('%Y-%m-%d')}",
                            body_style),
                        Paragraph(f"<b>Reason:</b> {sub.override_reason}", body_style)
                    ])
                if sub.display_score is not None:
                    total_score_achieved += sub.display_score;
                    total_max_score_possible += assign.max_score;
                    evaluated_assignments_count += 1
            else:
                story.append(Paragraph("<i>Pending Evaluation</i>", body_style))
            story.append(Spacer(1, 0.15 * inch))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"<b>Overall Performance Summary:</b>", header_style))
    if evaluated_assignments_count > 0:
        avg_percent = (total_score_achieved / total_max_score_possible) * 100 if total_max_score_possible > 0 else 0
        story.extend([
            Paragraph(
                f"Total Score: {total_score_achieved:.2f} / {total_max_score_possible:.2f} ({evaluated_assignments_count} evaluated)",
                body_style),
            Paragraph(f"Average Percentage: {avg_percent:.2f}%", body_style)
        ])
    else:
        story.append(Paragraph("No assignments evaluated yet.", body_style))
    doc.build(story);
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"student_{student.username}_report.pdf",
                     mimetype='application/pdf')


@app.route('/student/logout')
def student_logout():
    session.pop('student_logged_in', None);
    session.pop('student_id', None)
    flash('Student logged out successfully.', 'info');
    return redirect(url_for('index'))


# Combined Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'admin_logged_in' in session: return redirect(url_for('admin_dashboard'))
    if 'teacher_logged_in' in session: return redirect(url_for('teacher_dashboard'))
    if 'student_logged_in' in session: return redirect(url_for('student_dashboard'))
    default_role = request.args.get('role', 'student')
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        role = request.form.get('role', 'student')
        admin_code = request.form.get('admin_code', '')
        if not username or not password:
            flash('Username and password are required.', 'danger')
        else:
            hashed_password = hash_password(password)
            user = None
            if role == 'teacher':
                user = Teacher.query.filter_by(username=username, password=hashed_password).first()
            elif role == 'student':
                user = Student.query.filter_by(username=username, password=hashed_password).first()
            elif role == 'admin':
                if admin_code != os.environ.get('ADMIN_SECRET', 'admin123'):
                    flash('Invalid admin code.', 'danger')
                else:
                    user = Admin.query.filter_by(username=username, password=hashed_password).first()
            if user:
                session[f'{role}_logged_in'], session[f'{role}_id'] = True, user.id
                flash('Login successful!', 'success');
                return redirect(url_for(f'{role}_dashboard'))
            else:
                flash(f'Invalid credentials or role for {role}.', 'danger')
        return render_template('login.html', error='Login failed.', default_role=role, submitted_username=username)
    return render_template('login.html', default_role=default_role)


# Placeholder routes that now flash messages
@app.route('/teacher/report/classroom/<int:classroom_id>')
def classroom_report(classroom_id):
    if not session.get('teacher_logged_in'): return redirect(url_for('login', role='teacher'))
    classroom = Classroom.query.get_or_404(classroom_id)
    if classroom.teacher_id != session.get('teacher_id'):
        flash("Permission denied.", "danger");
        return redirect(url_for('teacher_dashboard'))
    flash(f"Classroom-specific PDF report for '{classroom.name}' is under development.", "info")
    return redirect(url_for('teacher_dashboard', active_tab='classroom-manager'))


@app.route('/teacher/submissions/assignment/<int:assignment_id>')
def view_submissions(assignment_id):
    if not session.get('teacher_logged_in'): return redirect(url_for('login', role='teacher'))
    assignment = Assignment.query.get_or_404(assignment_id)
    if assignment.teacher_id != session.get('teacher_id'):
        flash("Permission denied.", "danger");
        return redirect(url_for('teacher_dashboard'))
    flash(f"Displaying submissions for '{assignment.title}' in Evaluation Hub.", "info")
    return redirect(url_for('teacher_dashboard', filter_assignment_id=assignment_id, active_tab='evaluation-hub'))


@app.route('/teacher/submission/<int:submission_id>/view_details')
def view_submission_details(submission_id):
    if not session.get('teacher_logged_in'): return redirect(url_for('login', role='teacher'))
    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.get_or_404(submission.assignment_id)
    if assignment.teacher_id != session.get('teacher_id'):
        flash("Permission denied.", "danger");
        return redirect(url_for('teacher_dashboard'))
    flash(f"Detailed view for submission ID {submission.id} is under development. Key details are in Evaluation Hub.",
          "info")
    return redirect(url_for('teacher_dashboard', filter_assignment_id=assignment.id, active_tab='evaluation-hub'))


@app.route('/teacher/assignment/<int:assignment_id>/analytics')
def assignment_analytics(assignment_id):
    if not session.get('teacher_logged_in'): return redirect(url_for('login', role='teacher'))
    assignment = Assignment.query.get_or_404(assignment_id)
    if assignment.teacher_id != session.get('teacher_id'):
        flash("Permission denied.", "danger");
        return redirect(url_for('teacher_dashboard'))
    flash(f"Analytics for assignment '{assignment.title}' is under development.", "info")
    return redirect(url_for('teacher_dashboard', active_tab='reports'))


# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    flash('Error 404: The page you are looking for does not exist.', 'danger')
    target_dashboard = 'index'
    if 'teacher_logged_in' in session:
        target_dashboard = 'teacher_dashboard'
    elif 'student_logged_in' in session:
        target_dashboard = 'student_dashboard'
    elif 'admin_logged_in' in session:
        target_dashboard = 'admin_dashboard'
    return redirect(url_for(target_dashboard))


@app.errorhandler(SQLAlchemyError)
def handle_db_error(e):
    error_type = type(e).__name__;
    error_details = str(e)
    orig_error = getattr(e, 'orig', None)
    if orig_error: error_details += f" (Original error: {str(orig_error)})"
    full_error_message = f"Database Error ({error_type}): {error_details}"
    print(full_error_message, file=sys.stderr);
    app.logger.error(full_error_message)
    db.session.rollback()
    user_message = "A database operation failed."
    if isinstance(e, IntegrityError):
        user_message = "Failed to save: data conflicts with existing records (e.g., duplicate username/email or invalid reference). Check input."
    elif isinstance(e, DataError):
        user_message = "Failed to save: incorrect data format for the database."
    elif isinstance(e, OperationalError):
        user_message = "Database connection/operational error. Try again or contact support."
    flash_message = f'{user_message} (Details logged for admin)' if not app.debug else f'{user_message} Details: {full_error_message}'
    flash(flash_message, 'danger')
    target_dashboard = 'index'
    if 'teacher_logged_in' in session:
        target_dashboard = 'teacher_dashboard'
    elif 'student_logged_in' in session:
        target_dashboard = 'student_dashboard'
    elif 'admin_logged_in' in session:
        target_dashboard = 'admin_dashboard'
    return redirect(url_for(target_dashboard))


@app.errorhandler(500)
def internal_server_error(e):
    print(f"FATAL: Internal Server Error: {e}", file=sys.stderr)
    app.logger.error(f"Internal Server Error: {e}", exc_info=True)
    flash('Error 500: An unexpected internal server error occurred. Please try again or contact support.', 'danger')
    target_dashboard = 'index'
    if 'teacher_logged_in' in session:
        target_dashboard = 'teacher_dashboard'
    elif 'student_logged_in' in session:
        target_dashboard = 'student_dashboard'
    elif 'admin_logged_in' in session:
        target_dashboard = 'admin_dashboard'
    return redirect(url_for(target_dashboard))


# Main Execution
if __name__ == '__main__':
    print("Reminder: If models changed & using SQLite, delete 'exam_system.db' for dev schema reset (ERASES DATA).")
    with app.app_context():
        try:
            db.create_all();
            print("DB tables created/ensured.")
            if not Admin.query.first():
                db.session.add(
                    Admin(username='admin', password=hash_password(os.environ.get('DEFAULT_ADMIN_PASS', 'admin123'))))
                print(f"Default admin 'admin' created.")
            if not Teacher.query.first():
                teacher = Teacher(username='teacher',
                                  password=hash_password(os.environ.get('DEFAULT_TEACHER_PASS', 'teacher123')),
                                  full_name='Ada Lovelace', email='ada.lovelace@example.com')
                db.session.add(teacher);
                db.session.commit()
                print(f"Default teacher 'teacher' created.")
                if not Classroom.query.first():
                    classroom = Classroom(name='Intro to CS', teacher_id=teacher.id, department='Computer Science',
                                          year=1, description='Fundamentals.')
                    db.session.add(classroom)
                    db.session.commit()
                    print(
                        f"Default classroom '{classroom.name}' created for teacher '{teacher.username}'. Invite: {classroom.invite_code}")
            db.session.commit()
        except OperationalError as oe:
            print(f"CRITICAL DB ERROR: {oe}", file=sys.stderr); sys.exit(1)
        except Exception as e:
            print(f"Setup Error: {e}", file=sys.stderr); db.session.rollback()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

