package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.request.SendMessageRequest;
import com.example.Auto_BE.dto.response.ChatResponse;
import com.example.Auto_BE.dto.response.MessageResponse;
import com.example.Auto_BE.entity.Chat;
import com.example.Auto_BE.entity.Message;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.UserChat;
import com.example.Auto_BE.exception.ResourceNotFoundException;
import com.example.Auto_BE.repository.ChatRepository;
import com.example.Auto_BE.repository.MessageRepository;
import com.example.Auto_BE.repository.UserChatRepository;
import com.example.Auto_BE.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ChatService {
    
    private final ChatRepository chatRepository;
    private final MessageRepository messageRepository;
    private final UserRepository userRepository;
    private final UserChatRepository userChatRepository;
    private final SimpMessagingTemplate messagingTemplate;
    
    /**
     * Lấy tất cả chat của user
     */
    @Transactional(readOnly = true)
    public List<ChatResponse> getAllChats(Long userId) {
        List<UserChat> userChats = userChatRepository.findAllByUserId(userId);
        
        return userChats.stream().map(userChat -> {
            Chat chat = userChat.getChat();
            
            // Lấy thông tin người kia (cho chat 1-1)
            List<UserChat> chatMembers = userChatRepository.findAllByChatId(chat.getId());
            UserChat otherUserChat = chatMembers.stream()
                    .filter(uc -> !uc.getUser().getId().equals(userId))
                    .findFirst()
                    .orElse(null);
            
            User otherUser = otherUserChat != null ? otherUserChat.getUser() : null;
            
            // Build response
            ChatResponse.ChatResponseBuilder builder = ChatResponse.builder()
                    .id(chat.getId())
                    .lastMessage(chat.getLastMessageContent())
                    .lastMessageTime(chat.getLastMessageAt())
                    .unreadCount(userChat.getUnreadCount() != null ? userChat.getUnreadCount().longValue() : 0L)
                    .createdAt(chat.getCreatedAt())
                    .updatedAt(chat.getUpdatedAt());
            
            // Thêm thông tin users (cho chat 1-1, set user1 = current user, user2 = other user)
            if (otherUser != null) {
                builder.user1Id(userId)
                       .user1Name(userChat.getUser().getFullName())
                       .user1Avatar(userChat.getUser().getAvatar())
                       .user2Id(otherUser.getId())
                       .user2Name(otherUser.getFullName())
                       .user2Avatar(otherUser.getAvatar());
            }
            
            return builder.build();
        }).collect(Collectors.toList());
    }
    
    /**
     * Lấy chi tiết 1 chat
     */
    @Transactional(readOnly = true)
    public ChatResponse getChatById(Long chatId, Long userId) {
        // Kiểm tra user có trong chat không
        UserChat userChat = userChatRepository.findByUserIdAndChatId(userId, chatId)
                .orElseThrow(() -> new ResourceNotFoundException("Chat not found or access denied"));
        
        Chat chat = userChat.getChat();
        
        // Lấy thông tin người kia
        List<UserChat> chatMembers = userChatRepository.findAllByChatId(chatId);
        UserChat otherUserChat = chatMembers.stream()
                .filter(uc -> !uc.getUser().getId().equals(userId))
                .findFirst()
                .orElse(null);
        
        User otherUser = otherUserChat != null ? otherUserChat.getUser() : null;
        
        ChatResponse.ChatResponseBuilder builder = ChatResponse.builder()
                .id(chat.getId())
                .lastMessage(chat.getLastMessageContent())
                .lastMessageTime(chat.getLastMessageAt())
                .unreadCount(userChat.getUnreadCount() != null ? userChat.getUnreadCount().longValue() : 0L)
                .createdAt(chat.getCreatedAt())
                .updatedAt(chat.getUpdatedAt());
        
        if (otherUser != null) {
            builder.user1Id(userId)
                   .user1Name(userChat.getUser().getFullName())
                   .user1Avatar(userChat.getUser().getAvatar())
                   .user2Id(otherUser.getId())
                   .user2Name(otherUser.getFullName())
                   .user2Avatar(otherUser.getAvatar());
        }
        
        return builder.build();
    }
    
    /**
     * Lấy danh sách message trong chat (có phân trang)
     */
    @Transactional(readOnly = true)
    public Page<MessageResponse> getMessages(Long chatId, Long userId, Pageable pageable) {
        // Kiểm tra user có quyền xem chat này không
        userChatRepository.findByUserIdAndChatId(userId, chatId)
                .orElseThrow(() -> new ResourceNotFoundException("Chat not found or access denied"));
        
        Page<Message> messages = messageRepository.findByChatIdOrderByCreatedAtDesc(chatId, pageable);
        
        return messages.map(message -> MessageResponse.builder()
                .id(message.getId())
                .chatId(message.getChat().getId())
                .senderId(message.getSender().getId())
                .senderName(message.getSender().getFullName())
                .senderAvatar(message.getSender().getAvatar())
                .content(message.getContent())
                .isRead(message.getIsRead())
                .readAt(message.getReadAt())
                .createdAt(message.getCreatedAt())
                .build());
    }
    
    /**
     * Gửi tin nhắn (từ WebSocket hoặc REST API)
     */
    @Transactional
    public MessageResponse sendMessage(SendMessageRequest request, Long senderId) {
        User sender = userRepository.findById(senderId)
                .orElseThrow(() -> new ResourceNotFoundException("Sender not found"));
        
        Chat chat;
        
        // Nếu có chatId, dùng chat có sẵn
        if (request.getChatId() != null) {
            // Kiểm tra user có trong chat không
            UserChat senderUserChat = userChatRepository.findByUserIdAndChatId(senderId, request.getChatId())
                    .orElseThrow(() -> new ResourceNotFoundException("Chat not found or access denied"));
            chat = senderUserChat.getChat();
        } else {
            // Nếu không có chatId, tạo chat mới hoặc tìm chat 1-1 với receiverId
            User receiver = userRepository.findById(request.getReceiverId())
                    .orElseThrow(() -> new ResourceNotFoundException("Receiver not found"));
            
            chat = chatRepository.findDirectChatBetweenUsers(senderId, request.getReceiverId())
                    .orElseGet(() -> {
                        // Tạo chat mới
                        Chat newChat = new Chat();
                        newChat.setChatType("DIRECT");
                        newChat = chatRepository.save(newChat);
                        
                        // Tạo UserChat cho sender
                        UserChat senderUserChat = new UserChat();
                        senderUserChat.setChat(newChat);
                        senderUserChat.setUser(sender);
                        senderUserChat.setUnreadCount(0);
                        senderUserChat.setIsActive(true);
                        userChatRepository.save(senderUserChat);
                        
                        // Tạo UserChat cho receiver
                        UserChat receiverUserChat = new UserChat();
                        receiverUserChat.setChat(newChat);
                        receiverUserChat.setUser(receiver);
                        receiverUserChat.setUnreadCount(0);
                        receiverUserChat.setIsActive(true);
                        userChatRepository.save(receiverUserChat);
                        
                        return newChat;
                    });
        }
        
        // Tạo message mới
        Message message = new Message();
        message.setChat(chat);
        message.setSender(sender);
        message.setContent(request.getContent());
        message.setIsRead(false);
        message = messageRepository.save(message);
        
        // Cập nhật lastMessage của chat
        chat.setLastMessageContent(request.getContent());
        chat.setLastMessageAt(Instant.now());
        chatRepository.save(chat);
        
        // Tăng unreadCount của tất cả users khác trong chat
        List<UserChat> chatMembers = userChatRepository.findAllByChatId(chat.getId());
        for (UserChat userChat : chatMembers) {
            if (!userChat.getUser().getId().equals(senderId)) {
                userChat.setUnreadCount(userChat.getUnreadCount() + 1);
                userChatRepository.save(userChat);
            }
        }
        
        // Build response
        MessageResponse response = MessageResponse.builder()
                .id(message.getId())
                .chatId(chat.getId())
                .senderId(sender.getId())
                .senderName(sender.getFullName())
                .senderAvatar(sender.getAvatar())
                .content(message.getContent())
                .isRead(false)
                .createdAt(message.getCreatedAt())
                .build();
        
        // Gửi message qua WebSocket đến tất cả users khác trong chat
        for (UserChat userChat : chatMembers) {
            if (!userChat.getUser().getId().equals(senderId)) {
                messagingTemplate.convertAndSendToUser(
                        userChat.getUser().getEmail(),
                        "/queue/messages",
                        response
                );
            }
        }
        
        return response;
    }

    /**
     * Tạo hoặc lấy chat 1-1 giữa 2 users
     */
    @Transactional
    public ChatResponse createOrGetChat(Long user1Id, Long user2Id) {
        User user1 = userRepository.findById(user1Id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found"));
        User user2 = userRepository.findById(user2Id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found"));
        
        // Tìm chat đã tồn tại
        Chat chat = chatRepository.findDirectChatBetweenUsers(user1Id, user2Id)
                .orElseGet(() -> {
                    // Tạo chat mới
                    Chat newChat = new Chat();
                    newChat.setChatType("DIRECT");
                    newChat = chatRepository.save(newChat);
                    
                    // Tạo UserChat cho user1
                    UserChat userChat1 = new UserChat();
                    userChat1.setChat(newChat);
                    userChat1.setUser(user1);
                    userChat1.setUnreadCount(0);
                    userChat1.setIsActive(true);
                    userChatRepository.save(userChat1);
                    
                    // Tạo UserChat cho user2
                    UserChat userChat2 = new UserChat();
                    userChat2.setChat(newChat);
                    userChat2.setUser(user2);
                    userChat2.setUnreadCount(0);
                    userChat2.setIsActive(true);
                    userChatRepository.save(userChat2);
                    
                    return newChat;
                });
        
        // Build response
        return buildChatResponse(chat, user1Id);
    }

    /**
     * Xây dựng ChatResponse cho 1 chat theo user hiện tại (để trả về UI)
     */
    private ChatResponse buildChatResponse(Chat chat, Long userId) {
    // Tìm UserChat của user hiện tại trong chat
    UserChat currentUserChat = userChatRepository.findByUserIdAndChatId(userId, chat.getId())
        .orElse(null);

    // Lấy thông tin người kia (nếu có)
    List<UserChat> chatMembers = userChatRepository.findAllByChatId(chat.getId());
    UserChat otherUserChat = chatMembers.stream()
        .filter(uc -> !uc.getUser().getId().equals(userId))
        .findFirst()
        .orElse(null);

    User otherUser = otherUserChat != null ? otherUserChat.getUser() : null;

    ChatResponse.ChatResponseBuilder builder = ChatResponse.builder()
        .id(chat.getId())
        .lastMessage(chat.getLastMessageContent())
        .lastMessageTime(chat.getLastMessageAt())
        .unreadCount(currentUserChat != null && currentUserChat.getUnreadCount() != null ? currentUserChat.getUnreadCount().longValue() : 0L)
        .createdAt(chat.getCreatedAt())
        .updatedAt(chat.getUpdatedAt());

    if (otherUser != null && currentUserChat != null) {
        builder.user1Id(userId)
            .user1Name(currentUserChat.getUser().getFullName())
            .user1Avatar(currentUserChat.getUser().getAvatar())
            .user2Id(otherUser.getId())
            .user2Name(otherUser.getFullName())
            .user2Avatar(otherUser.getAvatar());
    }

    return builder.build();
    }
    
    /**
     * Đánh dấu tất cả message trong chat là đã đọc
     */
    @Transactional
    public void markMessagesAsRead(Long chatId, Long userId) {
        // Kiểm tra user có trong chat không
        userChatRepository.findByUserIdAndChatId(userId, chatId)
                .orElseThrow(() -> new ResourceNotFoundException("Chat not found or access denied"));
        
        // Đánh dấu messages là đã đọc
        messageRepository.markAllAsRead(chatId, userId);
        
        // Reset unread count
        userChatRepository.resetUnreadCount(userId, chatId);
    }
}