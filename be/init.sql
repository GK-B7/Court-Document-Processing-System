-- Database initialization for Document Processing API
-- Works with existing customers and actions tables

-- Enable UUID extension if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop only the new tables if they exist (NOT customers and actions)
DROP TABLE IF EXISTS processing_logs CASCADE;
DROP TABLE IF EXISTS review_queue CASCADE;
DROP TABLE IF EXISTS jobs CASCADE;

-- ─────────────────────────────────────────────────────────────────────────────
-- NOTE: customers and actions tables already exist - do not modify them
-- ─────────────────────────────────────────────────────────────────────────────

-- Verify existing tables structure (these should already exist)
-- customers table should have: national_id, customer_id
-- actions table should have: action_name, description

-- Create additional indexes on existing tables if they don't exist
DO $$ 
BEGIN
    -- Add indexes for customers table if they don't exist
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'customers' AND indexname = 'idx_customers_customer_id') THEN
        CREATE INDEX idx_customers_customer_id ON customers(customer_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'customers' AND indexname = 'idx_customers_national_id') THEN
        CREATE INDEX idx_customers_national_id ON customers(national_id);
    END IF;
    
    -- Add index for actions table if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'actions' AND indexname = 'idx_actions_action_name') THEN
        CREATE INDEX idx_actions_action_name ON actions(action_name);
    END IF;
END $$;

-- ─────────────────────────────────────────────────────────────────────────────
-- Jobs Table (For document processing workflow)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(100) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress DECIMAL(5,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    error_message TEXT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for jobs
CREATE INDEX idx_jobs_job_id ON jobs(job_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at);

-- ─────────────────────────────────────────────────────────────────────────────
-- Review Queue (For human review items)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE review_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(100) NOT NULL,
    national_id VARCHAR(20) NOT NULL,
    customer_id VARCHAR(20),
    original_action TEXT NOT NULL,
    matched_action VARCHAR(50),
    confidence DECIMAL(5,2),
    page_number INTEGER,
    context TEXT,
    review_reason TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    assigned_to VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP NULL,
    reviewed_by VARCHAR(100),
    review_decision VARCHAR(50),
    review_notes TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Foreign key constraints to existing tables
    CONSTRAINT fk_review_jobs FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
    -- Note: Cannot add FK to actions table without knowing exact structure
);

-- Create indexes for review queue
CREATE INDEX idx_review_job_id ON review_queue(job_id);
CREATE INDEX idx_review_status ON review_queue(status);
CREATE INDEX idx_review_priority ON review_queue(priority);
CREATE INDEX idx_review_created_at ON review_queue(created_at);
CREATE INDEX idx_review_national_id ON review_queue(national_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- Processing Logs (For audit trail and debugging)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE processing_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(100) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    log_level VARCHAR(20) DEFAULT 'INFO',
    message TEXT NOT NULL,
    data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    CONSTRAINT fk_logs_jobs FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

-- Create indexes for processing logs
CREATE INDEX idx_logs_job_id ON processing_logs(job_id);
CREATE INDEX idx_logs_agent_name ON processing_logs(agent_name);
CREATE INDEX idx_logs_log_level ON processing_logs(log_level);
CREATE INDEX idx_logs_created_at ON processing_logs(created_at);

-- ─────────────────────────────────────────────────────────────────────────────
-- Views for easier querying (using existing tables)
-- ─────────────────────────────────────────────────────────────────────────────

-- View for pending review items with customer information
CREATE VIEW pending_reviews AS
SELECT 
    rq.id,
    rq.job_id,
    rq.national_id,
    rq.customer_id,
    c.customer_id as verified_customer_id,
    rq.original_action,
    rq.matched_action,
    a.description as action_description,
    rq.confidence,
    rq.page_number,
    rq.context,
    rq.review_reason,
    rq.priority,
    rq.created_at
FROM review_queue rq
LEFT JOIN customers c ON rq.national_id::text = c.national_id::text
LEFT JOIN actions a ON rq.matched_action = a.action_name
WHERE rq.status = 'pending'
ORDER BY rq.priority ASC, rq.created_at ASC;

-- View for job statistics
CREATE VIEW job_statistics AS
SELECT 
    j.job_id,
    j.filename,
    j.status,
    j.progress,
    j.created_at,
    j.completed_at,
    EXTRACT(EPOCH FROM (COALESCE(j.completed_at, NOW()) - j.created_at)) as processing_time_seconds,
    COUNT(rq.id) as review_items_count,
    COUNT(CASE WHEN rq.status = 'pending' THEN 1 END) as pending_reviews,
    COUNT(CASE WHEN rq.status = 'approved' THEN 1 END) as approved_reviews,
    COUNT(CASE WHEN rq.status = 'rejected' THEN 1 END) as rejected_reviews
FROM jobs j
LEFT JOIN review_queue rq ON j.job_id = rq.job_id
GROUP BY j.job_id, j.filename, j.status, j.progress, j.created_at, j.completed_at;

-- ─────────────────────────────────────────────────────────────────────────────
-- Functions that work with existing tables
-- ─────────────────────────────────────────────────────────────────────────────

-- Function to get customer by national ID (works with existing customers table)
CREATE OR REPLACE FUNCTION get_customer_by_national_id(search_national_id TEXT)
RETURNS TABLE(customer_id VARCHAR, national_id TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT c.customer_id, c.national_id::text
    FROM customers c
    WHERE c.national_id::text = search_national_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get all actions from existing actions table
CREATE OR REPLACE FUNCTION get_supported_actions()
RETURNS TABLE(action_name VARCHAR, description TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT a.action_name::varchar, a.description
    FROM actions a
    ORDER BY a.action_name;
END;
$$ LANGUAGE plpgsql;

-- Function to create a new job
CREATE OR REPLACE FUNCTION create_job(
    p_job_id VARCHAR,
    p_filename VARCHAR,
    p_file_path VARCHAR
)
RETURNS UUID AS $$
DECLARE
    new_id UUID;
BEGIN
    INSERT INTO jobs (job_id, filename, file_path, status)
    VALUES (p_job_id, p_filename, p_file_path, 'pending')
    RETURNING id INTO new_id;
    
    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- Function to validate if action exists in your actions table
CREATE OR REPLACE FUNCTION is_valid_action(action_to_check VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS(SELECT 1 FROM actions WHERE action_name = action_to_check);
END;
$$ LANGUAGE plpgsql;

-- Function to get customer count
CREATE OR REPLACE FUNCTION get_customer_count()
RETURNS INTEGER AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM customers);
END;
$$ LANGUAGE plpgsql;

-- Function to get action count  
CREATE OR REPLACE FUNCTION get_action_count()
RETURNS INTEGER AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM actions);
END;
$$ LANGUAGE plpgsql;

-- ─────────────────────────────────────────────────────────────────────────────
-- Database permissions
-- ─────────────────────────────────────────────────────────────────────────────

-- Grant permissions to your database user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO postgres;

-- Create cleanup function for old records (optional)
CREATE OR REPLACE FUNCTION cleanup_old_records(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete old completed jobs and related records
    WITH deleted_jobs AS (
        DELETE FROM jobs 
        WHERE status IN ('completed', 'failed') 
        AND created_at < NOW() - INTERVAL '1 day' * days_to_keep
        RETURNING job_id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted_jobs;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ─────────────────────────────────────────────────────────────────────────────
-- Verification queries (to check existing data)
-- ─────────────────────────────────────────────────────────────────────────────

-- Display information about existing tables
DO $$
DECLARE
    customer_count INTEGER;
    action_count INTEGER;
BEGIN
    -- Get counts from existing tables
    SELECT COUNT(*) INTO customer_count FROM customers;
    SELECT COUNT(*) INTO action_count FROM actions;
    
    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE 'Working with existing tables:';
    RAISE NOTICE '  - customers table: % records found', customer_count;
    RAISE NOTICE '  - actions table: % records found', action_count;
    RAISE NOTICE 'New tables created: jobs, review_queue, processing_logs';
    RAISE NOTICE 'Views created: pending_reviews, job_statistics';
    RAISE NOTICE 'Functions created: get_customer_by_national_id, get_supported_actions, create_job, etc.';
END $$;

-- Show sample of existing data (first 3 records from each table)
SELECT 'EXISTING CUSTOMERS:' as info;
SELECT customer_id, national_id FROM customers LIMIT 3;

SELECT 'EXISTING ACTIONS:' as info;
SELECT action_name, description FROM actions LIMIT 10;
