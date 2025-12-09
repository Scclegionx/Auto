package com.example.Auto_BE.entity;

import com.example.Auto_BE.entity.enums.EBloodType;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.util.List;

@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Accessors(chain = true)
@Table(name = "elder_users")
@DiscriminatorValue("ELDER")
public class ElderUser extends User {
    
    @Column(name = "blood_type")
    @Enumerated(EnumType.STRING)
    private EBloodType bloodType;

    @Column(name = "height")
    private Double height; // Chiều cao tính bằng cm

    @Column(name = "weight")
    private Double weight; // Cân nặng tính bằng kg

    @OneToMany(mappedBy = "elderUser", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<EmergencyContact> emergencyContacts;

    @OneToMany(mappedBy = "elderUser", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<MedicationReminder> medicationReminders;

    @OneToMany(mappedBy = "elderUser", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<Prescriptions> prescriptions;

    @OneToMany(mappedBy = "elderUser", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<MedicalDocument> medicalDocuments;

    @OneToMany(mappedBy = "elderUser", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<ElderSupervisor> supervisorRelations;
}
