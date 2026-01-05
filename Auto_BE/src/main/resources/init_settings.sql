-- Script để khởi tạo các settings mặc định
-- Chạy script này sau khi tạo bảng settings và setting_users

-- Insert các settings mặc định
INSERT INTO settings (setting_key, name, description, default_value, possible_values, is_active, created_at, updated_at)
VALUES 
    ('theme', 'Nền', 'Chọn chế độ nền sáng hoặc tối', 'light', 'light,dark', true, NOW(), NOW()),
    ('font_size', 'Font Size', 'Kích thước chữ', '14', '10,12,14,16,18,20', true, NOW(), NOW()),
    ('voice_support', 'Hỗ trợ nói', 'Bật/tắt tính năng hỗ trợ giọng nói', 'on', 'on,off', true, NOW(), NOW());

-- Lưu ý: 
-- - Các giá trị trong possible_values được phân cách bởi dấu phẩy
-- - default_value phải nằm trong possible_values
-- - Khi có setting mới, chỉ cần thêm vào bảng settings này

