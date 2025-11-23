# Chat Feature Refactoring Summary

## 🎯 Mục tiêu
Refactor chat system từ thiết kế **2 foreign keys (user1, user2)** sang **Many-to-Many với bảng trung gian** để:
- ✅ Dễ vẽ ERD (Entity Relationship Diagram)
- ✅ Dễ mở rộng thành group chat
- ✅ Theo best practices của database design

---

## 📊 So sánh thiết kế

### Thiết kế cũ (2 Foreign Keys)
```
Chat
├── user1_id (FK) ──→ User
├── user2_id (FK) ──→ User
├── user1_unread_count
└── user2_unread_count
```
**Vấn đề**:
- ❌ Khó vẽ ERD (2 FK cùng trỏ đến 1 bảng)
- ❌ Không mở rộng được group chat
- ❌ Phải xác định user1 vs user2

### Thiết kế mới (Many-to-Many)
```
User ←→ UserChat ←→ Chat
         ├── user_id (FK)
         ├── chat_id (FK)
         └── unread_count
```
**Ưu điểm**:
- ✅ ERD chuẩn, dễ vẽ
- ✅ Dễ mở rộng group chat (>2 users)
- ✅ Mỗi user có unread_count riêng
- ✅ Có thể soft delete (is_active = false)

---

## 📁 Files đã tạo/sửa

### Entities Created:
1. **UserChat.java** (NEW) - Bảng trung gian
   - Fields: user_id, chat_id, unread_count, is_active
   - Unique constraint: (user_id, chat_id)

### Entities Modified:
2. **Chat.java** (REFACTORED)
   - Removed: user1, user2, user1UnreadCount, user2UnreadCount
   - Added: chatType, chatName, userChats (List<UserChat>)
   - Now supports both DIRECT (1-1) and GROUP chat

3. **Message.java** (NO CHANGE)
   - Still references Chat and User (sender)

### Repositories:
4. **UserChatRepository.java** (NEW)
   - `findAllByUserId()` - Lấy tất cả chats của user
   - `findByUserIdAndChatId()` - Check user có trong chat không
   - `findAllByChatId()` - Lấy tất cả users trong chat
   - `resetUnreadCount()` - Reset unread count

5. **ChatRepository.java** (REFACTORED)
   - Removed: findByUserIds(), findAllByUserId(), findByIdAndUserId()
   - Added: `findDirectChatBetweenUsers()` - Tìm chat 1-1 giữa 2 users

6. **MessageRepository.java** (NO CHANGE)

### Services:
7. **ChatService.java** (REFACTORED)
   - Updated all methods to work with UserChat
   - `getAllChats()` - Query từ UserChatRepository
   - `getChatById()` - Check qua UserChat
   - `sendMessage()` - Tạo UserChat cho sender & receiver
   - `markMessagesAsRead()` - Reset qua UserChatRepository

### Controllers:
8. **ChatController.java** (NO CHANGE)

### Resources:
9. **chat_tables_mysql.sql** (NEW) - Migration SQL script
10. **chat_erd.puml** (NEW) - PlantUML ERD diagram
11. **send_message_sequence.puml** (NEW) - Sequence diagram

### Documentation:
12. **CHAT_API_README.md** (UPDATED) - Updated ERD section

---

## 🗄️ Database Schema

### Tables:
```sql
-- 3 tables chính:
chats              -- Chat conversations
user_chats         -- Many-to-Many junction table
messages           -- Chat messages

-- Foreign Keys:
user_chats.user_id  → users.id
user_chats.chat_id  → chats.id
messages.chat_id    → chats.id
messages.sender_id  → users.id
```

### ERD:
```
User ──┬─→ UserChat ──┬─→ Chat ──→ Message
       │              │
       │              │
       └──────────────┴───→ Message (sender)
```

---

## 🔄 Migration Guide

### Nếu đã có data cũ:
```sql
-- 1. Tạo bảng mới
CREATE TABLE chats_new (...);
CREATE TABLE user_chats (...);

-- 2. Migrate data
INSERT INTO chats_new (id, chat_type, last_message_content, last_message_at, created_at, updated_at)
SELECT id, 'DIRECT', last_message_content, last_message_at, created_at, updated_at
FROM chats;

INSERT INTO user_chats (user_id, chat_id, unread_count, is_active)
SELECT user1_id, id, user1_unread_count, TRUE FROM chats
UNION ALL
SELECT user2_id, id, user2_unread_count, TRUE FROM chats;

-- 3. Drop old table
DROP TABLE chats;
RENAME TABLE chats_new TO chats;
```

### Nếu fresh install:
```bash
# Chỉ cần run file migration
mysql -u root -p database_name < src/main/resources/chat_tables_mysql.sql
```

---

## 📈 API Changes

### API Endpoints (NO CHANGE):
```
GET    /api/chat                  - Lấy danh sách chats
GET    /api/chat/{id}             - Chi tiết chat
GET    /api/chat/{id}/messages    - Lấy messages
POST   /api/chat/send             - Gửi tin nhắn
PUT    /api/chat/{id}/read        - Đánh dấu đã đọc
```

### WebSocket (NO CHANGE):
```
Connect:    /ws/chat
Send to:    /app/chat.send
Receive on: /user/queue/messages
```

### Response Structure (SLIGHT CHANGE):
```json
{
  "id": 1,
  "user1Id": 1,        // Luôn là current user
  "user1Name": "...",
  "user1Avatar": "...",
  "user2Id": 2,        // Người kia (cho chat 1-1)
  "user2Name": "...",
  "user2Avatar": "...",
  "lastMessage": "...",
  "lastMessageTime": "...",
  "unreadCount": 3     // Unread count của current user
}
```

---

## ✅ Testing Checklist

- [ ] Run migration SQL
- [ ] Test create new chat
- [ ] Test send message
- [ ] Test receive message via WebSocket
- [ ] Test unread count increment
- [ ] Test mark as read
- [ ] Test get chat list
- [ ] Verify ERD matches implementation
- [ ] Test với >2 messages in a chat
- [ ] Test concurrent users

---

## 🎨 Diagrams cho báo cáo

### ERD (PlantUML):
```bash
# Generate diagram
java -jar plantuml.jar src/main/resources/diagrams/chat_erd.puml
```
Output: `chat_erd.png`

### Sequence Diagram:
```bash
java -jar plantuml.jar src/main/resources/diagrams/send_message_sequence.puml
```
Output: `send_message_sequence.png`

---

## 🚀 Next Steps

1. **Backend**:
   - [ ] Run migration
   - [ ] Test APIs
   - [ ] Verify WebSocket works

2. **Frontend** (Android):
   - [ ] Update API calls (no change needed)
   - [ ] Update UI to show correct user info
   - [ ] Test real-time messaging

3. **Documentation**:
   - [ ] Thêm ERD vào báo cáo
   - [ ] Thêm sequence diagram
   - [ ] Giải thích thiết kế Many-to-Many

---

## 📝 Notes

- Vẫn là **chat 1-1** (DIRECT), chỉ thay đổi cách lưu trữ
- Dễ upgrade thành **group chat** sau này bằng cách:
  - Thêm >2 UserChat records cho 1 Chat
  - Set chatType = 'GROUP'
  - Set chatName cho group

- **Unread count** giờ lưu trong `user_chats`, mỗi user có unread count riêng
- **is_active** field cho phép soft delete (user rời chat)

---

## 🎓 Lý do refactor (cho báo cáo)

> "Trong thiết kế ban đầu, bảng Chat có 2 foreign keys (user1_id, user2_id) trỏ đến bảng User. 
> Thiết kế này gây khó khăn khi vẽ Entity Relationship Diagram (ERD) vì vi phạm nguyên tắc 
> chuẩn hóa cơ sở dữ liệu và không mở rộng được cho chat nhóm.
> 
> Sau khi refactor sang mô hình Many-to-Many với bảng trung gian UserChat, hệ thống trở nên 
> dễ hiểu hơn, tuân thủ best practices, và sẵn sàng cho các tính năng mở rộng trong tương lai."

