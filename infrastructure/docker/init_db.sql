-- Initialize AI Tutor Database Schema

-- Create students table
CREATE TABLE IF NOT EXISTS students (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    knowledge_level VARCHAR(50) DEFAULT 'Intermediate',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create modules table
CREATE TABLE IF NOT EXISTS modules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module_number INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    content_url TEXT,
    order_index INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create student_progress table
CREATE TABLE IF NOT EXISTS student_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    module_id UUID REFERENCES modules(id) ON DELETE CASCADE,
    mastery_score DECIMAL(5,2) DEFAULT 0.0,
    completed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(student_id, module_id)
);

-- Create interactions table (for tracking Q&A)
CREATE TABLE IF NOT EXISTS interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    topic VARCHAR(255),
    confidence_score DECIMAL(5,2),
    helpful BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create case_studies table
CREATE TABLE IF NOT EXISTS case_studies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    content TEXT NOT NULL,
    rubric JSONB,
    module_id UUID REFERENCES modules(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create case_submissions table
CREATE TABLE IF NOT EXISTS case_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    case_study_id UUID REFERENCES case_studies(id) ON DELETE CASCADE,
    analysis TEXT NOT NULL,
    score INTEGER,
    feedback TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    graded_at TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_student_external_id ON students(external_id);
CREATE INDEX idx_student_progress_student ON student_progress(student_id);
CREATE INDEX idx_interactions_student ON interactions(student_id);
CREATE INDEX idx_interactions_created ON interactions(created_at);
CREATE INDEX idx_case_submissions_student ON case_submissions(student_id);

-- Insert sample modules
INSERT INTO modules (module_number, title, description, order_index) VALUES
(1, 'AI Strategy Fundamentals', 'Introduction to AI strategy and its role in business transformation', 1),
(2, 'Organizational AI Adoption', 'Understanding how organizations adopt and implement AI initiatives', 2),
(3, 'Competitive Advantage through AI', 'Leveraging AI for strategic competitive advantage', 3),
(4, 'AI Ethics and Governance', 'Ethical considerations and governance frameworks for AI', 4),
(5, 'AI Implementation Frameworks', 'Practical frameworks for implementing AI strategies', 5);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
CREATE TRIGGER update_students_updated_at BEFORE UPDATE ON students
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_student_progress_updated_at BEFORE UPDATE ON student_progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (for development)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO tutor_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO tutor_admin;
