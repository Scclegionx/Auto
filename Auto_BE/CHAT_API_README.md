# Chat Feature - API Documentation

## Overview
Hệ thống chat 1-1 real-time sử dụng WebSocket (STOMP) và REST API.

## Authentication
Tất cả requests đều cần JWT token trong header:
```
Authorization: Bearer <token>
```

**Lưu ý**: JWT token cần có claim `userId` để hoạt động. Cần update code login để thêm userId vào token khi generate.

---

## REST API Endpoints

### 1. Lấy danh sách chat
```http
GET /api/chat
```
**Response:**
```json
[
  {
    "id": 1,
    "user1Id": 1,
    "user1Name": "Nguyễn Văn A",
    "user1Avatar": "https://...",
    "user2Id": 2,
    "user2Name": "Trần Thị B",
    "user2Avatar": "https://...",
    "lastMessage": "Xin chào",
    "lastMessageTime": "2024-01-15T10:30:00Z",
    "unreadCount": 3,
    "createdAt": "2024-01-15T09:00:00Z",
    "updatedAt": "2024-01-15T10:30:00Z"
  }
]
```

### 2. Lấy chi tiết chat
```http
GET /api/chat/{chatId}
```

### 3. Lấy danh sách tin nhắn
```http
GET /api/chat/{chatId}/messages?page=0&size=50
```
**Response:**
```json
{
  "content": [
    {
      "id": 1,
      "chatId": 1,
      "senderId": 2,
      "senderName": "Trần Thị B",
      "senderAvatar": "https://...",
      "content": "Xin chào",
      "isRead": true,
      "readAt": "2024-01-15T10:31:00Z",
      "createdAt": "2024-01-15T10:30:00Z"
    }
  ],
  "totalPages": 1,
  "totalElements": 10
}
```

### 4. Gửi tin nhắn (REST API)
```http
POST /api/chat/send
Content-Type: application/json

{
  "chatId": 1,          // Optional: nếu đã có chat
  "receiverId": 2,      // Optional: nếu tạo chat mới
  "content": "Hello"
}
```

### 5. Đánh dấu đã đọc
```http
PUT /api/chat/{chatId}/read
```

---

## WebSocket API

### Connection
```javascript
// Connect với SockJS + STOMP
const socket = new SockJS('http://localhost:8080/ws/chat', null, {
  transports: ['websocket', 'xhr-streaming', 'xhr-polling']
});

const stompClient = Stomp.over(socket);

// QUAN TRỌNG: Truyền JWT token qua header khi connect
stompClient.connect(
  { Authorization: 'Bearer ' + token },  // JWT token
  onConnected,
  onError
);
```

### Subscribe to receive messages
```javascript
function onConnected() {
  // Subscribe để nhận tin nhắn
  stompClient.subscribe('/user/queue/messages', onMessageReceived);
}

function onMessageReceived(payload) {
  const message = JSON.parse(payload.body);
  console.log('Received:', message);
  // message structure giống MessageResponse
}
```

### Send message
```javascript
function sendMessage(chatId, receiverId, content) {
  stompClient.send('/app/chat.send', {}, JSON.stringify({
    chatId: chatId,        // null nếu tạo chat mới
    receiverId: receiverId, // null nếu dùng chatId
    content: content
  }));
}
```

---

## Database Schema

### Chat Table (1-1 hoặc group)
```sql
CREATE TABLE chats (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  chat_type VARCHAR(20) NOT NULL DEFAULT 'DIRECT', -- DIRECT hoặc GROUP
  chat_name VARCHAR(100), -- Tên chat (null cho 1-1, có giá trị cho group)
  last_message_content TEXT,
  last_message_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### UserChat Table (Bảng trung gian Many-to-Many)
```sql
CREATE TABLE user_chats (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  chat_id BIGINT NOT NULL,
  unread_count INT DEFAULT 0, -- Số tin chưa đọc của user này
  is_active BOOLEAN DEFAULT TRUE, -- User còn trong chat không
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (chat_id) REFERENCES chats(id),
  UNIQUE KEY unique_user_chat (user_id, chat_id)
);
```

### Message Table
```sql
CREATE TABLE messages (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  chat_id BIGINT NOT NULL,
  sender_id BIGINT NOT NULL,
  content TEXT NOT NULL,
  is_read BOOLEAN DEFAULT FALSE,
  read_at TIMESTAMP NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (chat_id) REFERENCES chats(id),
  FOREIGN KEY (sender_id) REFERENCES users(id),
  INDEX idx_chat_created (chat_id, created_at DESC)
);
```

### ERD (Entity Relationship Diagram)
```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   User      │         │  UserChat    │         │   Chat      │
├─────────────┤         ├──────────────┤         ├─────────────┤
│ id (PK)     │────┐    │ id (PK)      │    ┌────│ id (PK)     │
│ email       │    │    │ user_id (FK) │    │    │ chat_type   │
│ full_name   │    └───→│ chat_id (FK) │←───┘    │ chat_name   │
│ avatar      │         │ unread_count │         │ last_msg... │
│ ...         │         │ is_active    │         │ last_msg... │
└─────────────┘         │ ...          │         └─────────────┘
                        └──────────────┘                │
                                                        │
                        ┌──────────────┐                │
                        │  Message     │                │
                        ├──────────────┤                │
                        │ id (PK)      │                │
                        │ chat_id (FK) │────────────────┘
                        │ sender_id(FK)│
                        │ content      │
                        │ is_read      │
                        │ read_at      │
                        └──────────────┘
```

**Quan hệ**:
- **User ←→ Chat**: Many-to-Many qua bảng trung gian `UserChat`
- **Chat → Message**: One-to-Many (1 chat có nhiều messages)
- **User → Message**: One-to-Many (1 user gửi nhiều messages)

**Ưu điểm của thiết kế này**:
- ✅ Dễ vẽ ERD (quan hệ Many-to-Many chuẩn)
- ✅ Dễ mở rộng thành group chat (>2 users)
- ✅ Mỗi user có unread count riêng trong `user_chats`
- ✅ Có thể thêm role (admin, member) vào `user_chats` sau này
- ✅ Có thể soft delete (user rời chat bằng cách set `is_active = false`)

---

## Flow Diagram

### Send Message Flow
```
1. Client gửi tin nhắn qua WebSocket hoặc REST API
   ↓
2. Server validate JWT token
   ↓
3. Tìm hoặc tạo Chat record (nếu chưa có)
   ↓
4. Lưu Message vào database
   ↓
5. Cập nhật lastMessage và unreadCount trong Chat
   ↓
6. Server gửi message đến người nhận qua WebSocket:
   /user/{receiverEmail}/queue/messages
   ↓
7. Client nhận và hiển thị message
```

### Read Message Flow
```
1. Client gọi PUT /api/chat/{chatId}/read
   ↓
2. Server đánh dấu tất cả messages trong chat là đã đọc
   ↓
3. Reset unreadCount về 0
```

---

## TODO: Update Login Code

**QUAN TRỌNG**: Cần update code login để thêm `userId` vào JWT token:

```java
// Trong AuthController hoặc LoginService
User user = userRepository.findByEmail(email).orElseThrow(...);
String token = jwtUtils.generateAccessToken(email, user.getId()); // Thêm userId
```

---

## Testing

### 1. Test REST API với Postman/cURL:
```bash
# Get all chats
curl -H "Authorization: Bearer <token>" http://localhost:8080/api/chat

# Send message
curl -X POST -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{"receiverId": 2, "content": "Hello"}' \
     http://localhost:8080/api/chat/send
```

### 2. Test WebSocket với Browser Console:
```javascript
// 1. Include SockJS và STOMP libraries
<script src="https://cdn.jsdelivr.net/npm/sockjs-client@1/dist/sockjs.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/stompjs@2.3.3/lib/stomp.min.js"></script>

// 2. Connect
const token = 'your-jwt-token';
const socket = new SockJS('http://localhost:8080/ws/chat');
const stompClient = Stomp.over(socket);

stompClient.connect(
  { Authorization: 'Bearer ' + token },
  () => {
    console.log('Connected!');
    stompClient.subscribe('/user/queue/messages', (msg) => {
      console.log('Received:', JSON.parse(msg.body));
    });
  },
  (error) => {
    console.error('Connection error:', error);
  }
);

// 3. Send message
stompClient.send('/app/chat.send', {}, JSON.stringify({
  receiverId: 2,
  content: 'Test message'
}));
```

---

## Dependencies Added

```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-websocket</artifactId>
</dependency>
```

---

## Security Notes

- JWT token được validate trong `HandshakeInterceptor` khi WebSocket handshake
- Token phải có trong header `Authorization: Bearer <token>`
- Không dùng query param để truyền token (kém bảo mật hơn)
- User chỉ có thể xem/gửi message trong chat mà họ tham gia
- Unread count được tính riêng cho từng user

---

## Frontend Integration (Android Kotlin)

Sẽ cần thêm dependency:
```kotlin
// build.gradle.kts
implementation("org.java-websocket:Java-WebSocket:1.5.3")
// hoặc
implementation("com.squareup.okhttp3:okhttp:4.11.0")
```

Example WebSocket client sẽ tạo sau khi backend hoàn chỉnh.
