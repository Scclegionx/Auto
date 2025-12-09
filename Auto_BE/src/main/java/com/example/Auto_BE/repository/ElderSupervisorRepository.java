package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.ElderSupervisor;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ElderSupervisorRepository extends JpaRepository<ElderSupervisor, Long> {

    // ===== Active Relationships (ACCEPTED) =====
    
    @Query("SELECT es FROM ElderSupervisor es WHERE es.elderUser.id = :elderUserId AND es.isActive = true AND es.status = 'ACCEPTED'")
    List<ElderSupervisor> findActiveByElderUserId(@Param("elderUserId") Long elderUserId);

    @Query("SELECT es FROM ElderSupervisor es WHERE es.supervisorUser.id = :supervisorUserId AND es.isActive = true AND es.status = 'ACCEPTED'")
    List<ElderSupervisor> findActiveBySupervisorUserId(@Param("supervisorUserId") Long supervisorUserId);

    @Query("SELECT es FROM ElderSupervisor es WHERE es.elderUser.id = :elderUserId AND es.supervisorUser.id = :supervisorUserId")
    Optional<ElderSupervisor> findByElderAndSupervisor(@Param("elderUserId") Long elderUserId, @Param("supervisorUserId") Long supervisorUserId);

    @Query("SELECT es FROM ElderSupervisor es " +
           "WHERE es.supervisorUser.id = :supervisorUserId " +
           "AND es.elderUser.id = :elderUserId " +
           "AND es.canViewPrescription = true " +
           "AND es.isActive = true " +
           "AND es.status = 'ACCEPTED'")
    Optional<ElderSupervisor> findActiveWithViewPermission(@Param("supervisorUserId") Long supervisorUserId, @Param("elderUserId") Long elderUserId);

    @Query("SELECT es FROM ElderSupervisor es " +
           "WHERE es.supervisorUser.id = :supervisorUserId " +
           "AND es.elderUser.id = :elderUserId " +
           "AND es.canUpdatePrescription = true " +
           "AND es.isActive = true " +
           "AND es.status = 'ACCEPTED'")
    Optional<ElderSupervisor> findActiveWithUpdatePermission(@Param("supervisorUserId") Long supervisorUserId, @Param("elderUserId") Long elderUserId);
    
    // ===== Request Management =====
    
    /**
     * Kiểm tra đã có request pending giữa elder và supervisor chưa
     */
    @Query("SELECT CASE WHEN COUNT(es) > 0 THEN true ELSE false END FROM ElderSupervisor es " +
           "WHERE es.elderUser.id = :elderUserId " +
           "AND es.supervisorUser.id = :supervisorUserId " +
           "AND es.status = 'PENDING'")
    boolean existsPendingRequest(@Param("elderUserId") Long elderUserId, @Param("supervisorUserId") Long supervisorUserId);
    
    /**
     * Kiểm tra đã có relationship accepted chưa
     */
    @Query("SELECT CASE WHEN COUNT(es) > 0 THEN true ELSE false END FROM ElderSupervisor es " +
           "WHERE es.elderUser.id = :elderUserId " +
           "AND es.supervisorUser.id = :supervisorUserId " +
           "AND es.status = 'ACCEPTED' " +
           "AND es.isActive = true")
    boolean existsAcceptedRelationship(@Param("elderUserId") Long elderUserId, @Param("supervisorUserId") Long supervisorUserId);
    
    /**
     * Lấy tất cả pending requests mà user nhận được (là receiver)
     * - Nếu userId là Elder → lấy requests từ Supervisors
     * - Nếu userId là Supervisor → lấy requests từ Elders
     */
    @Query("SELECT es FROM ElderSupervisor es " +
           "WHERE ((es.elderUser.id = :userId AND es.requester.id != :userId) " +
           "OR (es.supervisorUser.id = :userId AND es.requester.id != :userId)) " +
           "AND es.status = 'PENDING' " +
           "ORDER BY es.createdAt DESC")
    List<ElderSupervisor> findPendingReceivedRequests(@Param("userId") Long userId);
    
    /**
     * Lấy tất cả requests mà user đã gửi (là sender)
     */
    @Query("SELECT es FROM ElderSupervisor es " +
           "WHERE es.requester.id = :userId " +
           "ORDER BY es.createdAt DESC")
    List<ElderSupervisor> findSentRequestsByUserId(@Param("userId") Long userId);
    
    /**
     * Lấy pending requests mà user đã gửi
     */
    @Query("SELECT es FROM ElderSupervisor es " +
           "WHERE es.requester.id = :userId " +
           "AND es.status = 'PENDING' " +
           "ORDER BY es.createdAt DESC")
    List<ElderSupervisor> findPendingSentRequests(@Param("userId") Long userId);
}
