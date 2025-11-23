-- Migration: Chat feature với Many-to-Many relationship
-- Author: Auto
-- Date: 2025-11-17

-- 1. Tạo bảng chats
CREATE TABLE IF NOT EXISTS chats (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    chat_type VARCHAR(20) NOT NULL DEFAULT 'DIRECT' COMMENT 'DIRECT (1-1) hoặc GROUP (nhóm)',
    chat_name VARCHAR(100) COMMENT 'Tên chat (null cho 1-1, có giá trị cho group)',
    last_message_content VARCHAR(500) COMMENT 'Preview message cuối',
    last_message_at TIMESTAMP NULL COMMENT 'Thời gian message cuối',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_chat_type (chat_type),
    INDEX idx_last_message_at (last_message_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Bảng lưu chat conversations (1-1 hoặc group)';

-- 2. Tạo bảng trung gian user_chats (Many-to-Many)
CREATE TABLE IF NOT EXISTS user_chats (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL COMMENT 'ID của user',
    chat_id BIGINT NOT NULL COMMENT 'ID của chat',
    unread_count INT NOT NULL DEFAULT 0 COMMENT 'Số tin chưa đọc của user này',
    is_active BOOLEAN NOT NULL DEFAULT TRUE COMMENT 'User còn trong chat không',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_chat (user_id, chat_id),
    INDEX idx_user_id (user_id),
    INDEX idx_chat_id (chat_id),
    INDEX idx_user_active (user_id, is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Bảng trung gian Many-to-Many giữa User và Chat';

-- 3. Tạo bảng messages
CREATE TABLE IF NOT EXISTS messages (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    chat_id BIGINT NOT NULL COMMENT 'ID của chat',
    sender_id BIGINT NOT NULL COMMENT 'ID của người gửi',
    content TEXT NOT NULL COMMENT 'Nội dung tin nhắn',
    is_read BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Đã đọc chưa',
    read_at TIMESTAMP NULL COMMENT 'Thời gian đọc',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
    FOREIGN KEY (sender_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_chat_created (chat_id, created_at DESC),
    INDEX idx_sender (sender_id),
    INDEX idx_is_read (is_read)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Bảng lưu tin nhắn trong chat';

-- 4. Sample data (optional - for testing)
-- Tạo 1 chat 1-1 giữa user 1 và user 2
INSERT INTO chats (chat_type, chat_name, last_message_content, last_message_at) 
VALUES ('DIRECT', NULL, 'Xin chào!', NOW());

-- Thêm 2 users vào chat
INSERT INTO user_chats (user_id, chat_id, unread_count, is_active) 
VALUES 
    (1, LAST_INSERT_ID(), 0, TRUE),
    (2, LAST_INSERT_ID(), 1, TRUE);

-- Thêm 1 message
INSERT INTO messages (chat_id, sender_id, content, is_read) 
VALUES (LAST_INSERT_ID(), 1, 'Xin chào!', FALSE);
