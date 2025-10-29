-- ========================================
-- Migration: Update notifications table for grouped notifications
-- Date: 2025-10-29
-- ========================================

-- Step 1: Drop foreign key constraint (if exists)
SET @foreign_key_name = (
    SELECT CONSTRAINT_NAME 
    FROM information_schema.KEY_COLUMN_USAGE 
    WHERE TABLE_SCHEMA = DATABASE() 
    AND TABLE_NAME = 'notifications' 
    AND COLUMN_NAME = 'medication_reminder_id'
    LIMIT 1
);

SET @drop_fk_query = IF(@foreign_key_name IS NOT NULL,
    CONCAT('ALTER TABLE notifications DROP FOREIGN KEY ', @foreign_key_name),
    'SELECT 1');

PREPARE stmt FROM @drop_fk_query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Step 2: Add new columns (check if exists first)
-- Add title column
SET @col_exists = (SELECT COUNT(*) FROM information_schema.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND COLUMN_NAME = 'title');
SET @add_title = IF(@col_exists = 0, 
    'ALTER TABLE notifications ADD COLUMN title VARCHAR(255)', 
    'SELECT 1');
PREPARE stmt FROM @add_title;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add body column
SET @col_exists = (SELECT COUNT(*) FROM information_schema.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND COLUMN_NAME = 'body');
SET @add_body = IF(@col_exists = 0, 
    'ALTER TABLE notifications ADD COLUMN body VARCHAR(1000)', 
    'SELECT 1');
PREPARE stmt FROM @add_body;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add medication_count column
SET @col_exists = (SELECT COUNT(*) FROM information_schema.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND COLUMN_NAME = 'medication_count');
SET @add_count = IF(@col_exists = 0, 
    'ALTER TABLE notifications ADD COLUMN medication_count INT NOT NULL DEFAULT 1', 
    'SELECT 1');
PREPARE stmt FROM @add_count;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add medication_ids column
SET @col_exists = (SELECT COUNT(*) FROM information_schema.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND COLUMN_NAME = 'medication_ids');
SET @add_ids = IF(@col_exists = 0, 
    'ALTER TABLE notifications ADD COLUMN medication_ids VARCHAR(500)', 
    'SELECT 1');
PREPARE stmt FROM @add_ids;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add medication_names column
SET @col_exists = (SELECT COUNT(*) FROM information_schema.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND COLUMN_NAME = 'medication_names');
SET @add_names = IF(@col_exists = 0, 
    'ALTER TABLE notifications ADD COLUMN medication_names VARCHAR(1000)', 
    'SELECT 1');
PREPARE stmt FROM @add_names;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Step 3: Update existing data to new enum values
UPDATE notifications 
SET status = CASE 
    WHEN status = 'PENDING' THEN 'SENT'
    WHEN status = 'TAKEN' THEN 'SENT'
    WHEN status = 'MISSED' THEN 'FAILED'
    ELSE status
END
WHERE status IN ('PENDING', 'TAKEN', 'MISSED');

-- Step 4: Modify status column to support new enum values
ALTER TABLE notifications 
MODIFY COLUMN status VARCHAR(20) NOT NULL DEFAULT 'SENT';

-- Step 5: Drop old columns (if exist)
-- Drop retry_count
SET @col_exists = (SELECT COUNT(*) FROM information_schema.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND COLUMN_NAME = 'retry_count');
SET @drop_retry = IF(@col_exists > 0, 
    'ALTER TABLE notifications DROP COLUMN retry_count', 
    'SELECT 1');
PREPARE stmt FROM @drop_retry;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Drop medication_reminder_id
SET @col_exists = (SELECT COUNT(*) FROM information_schema.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND COLUMN_NAME = 'medication_reminder_id');
SET @drop_med_id = IF(@col_exists > 0, 
    'ALTER TABLE notifications DROP COLUMN medication_reminder_id', 
    'SELECT 1');
PREPARE stmt FROM @drop_med_id;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Step 6: Create indexes for better query performance
-- Index on reminder_time
SET @index_exists = (SELECT COUNT(*) FROM information_schema.STATISTICS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND INDEX_NAME = 'idx_notifications_reminder_time');
SET @create_idx1 = IF(@index_exists = 0, 
    'CREATE INDEX idx_notifications_reminder_time ON notifications(reminder_time)', 
    'SELECT 1');
PREPARE stmt FROM @create_idx1;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Index on user_id, status
SET @index_exists = (SELECT COUNT(*) FROM information_schema.STATISTICS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND INDEX_NAME = 'idx_notifications_user_status');
SET @create_idx2 = IF(@index_exists = 0, 
    'CREATE INDEX idx_notifications_user_status ON notifications(user_id, status)', 
    'SELECT 1');
PREPARE stmt FROM @create_idx2;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Index on user_id, reminder_time
SET @index_exists = (SELECT COUNT(*) FROM information_schema.STATISTICS 
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'notifications' AND INDEX_NAME = 'idx_notifications_user_time');
SET @create_idx3 = IF(@index_exists = 0, 
    'CREATE INDEX idx_notifications_user_time ON notifications(user_id, reminder_time DESC)', 
    'SELECT 1');
PREPARE stmt FROM @create_idx3;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Step 7: Verify changes
SELECT 'Migration completed successfully' AS status;

