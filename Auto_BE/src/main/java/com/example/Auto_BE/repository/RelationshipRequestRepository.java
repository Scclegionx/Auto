package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.RelationshipRequest;
import com.example.Auto_BE.entity.SupervisorUser;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.enums.ERelationshipRequestStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface RelationshipRequestRepository extends JpaRepository<RelationshipRequest, Long> {
    
    /**
     * Tìm request pending giữa Elder và Supervisor
     */
    Optional<RelationshipRequest> findByElderUserAndSupervisorUserAndStatus(
        ElderUser elderUser, 
        SupervisorUser supervisorUser, 
        ERelationshipRequestStatus status
    );
    
    /**
     * Kiểm tra đã có request pending giữa 2 users chưa
     */
    @Query("SELECT CASE WHEN COUNT(r) > 0 THEN true ELSE false END FROM RelationshipRequest r " +
           "WHERE r.elderUser = :elderUser AND r.supervisorUser = :supervisorUser AND r.status = 'PENDING'")
    boolean existsPendingRequest(
        @Param("elderUser") ElderUser elderUser,
        @Param("supervisorUser") SupervisorUser supervisorUser
    );
    
    /**
     * Lấy tất cả requests mà user là receiver (người nhận)
     * - Nếu user là Elder → lấy requests mà Supervisor gửi
     * - Nếu user là Supervisor → lấy requests mà Elder gửi
     */
    @Query("SELECT r FROM RelationshipRequest r " +
           "WHERE (r.elderUser.id = :userId AND r.requester.id != :userId) " +
           "OR (r.supervisorUser.id = :userId AND r.requester.id != :userId) " +
           "ORDER BY r.createdAt DESC")
    List<RelationshipRequest> findReceivedRequestsByUserId(@Param("userId") Long userId);
    
    /**
     * Lấy tất cả requests mà user là sender (người gửi)
     */
    @Query("SELECT r FROM RelationshipRequest r " +
           "WHERE r.requester.id = :userId " +
           "ORDER BY r.createdAt DESC")
    List<RelationshipRequest> findSentRequestsByUserId(@Param("userId") Long userId);
    
    /**
     * Lấy pending requests nhận được
     */
    @Query("SELECT r FROM RelationshipRequest r " +
           "WHERE ((r.elderUser.id = :userId AND r.requester.id != :userId) " +
           "OR (r.supervisorUser.id = :userId AND r.requester.id != :userId)) " +
           "AND r.status = 'PENDING' " +
           "ORDER BY r.createdAt DESC")
    List<RelationshipRequest> findPendingReceivedRequests(@Param("userId") Long userId);
}
