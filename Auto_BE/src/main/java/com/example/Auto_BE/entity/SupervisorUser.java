package com.example.Auto_BE.entity;

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
@Table(name = "supervisor_users")
@DiscriminatorValue("SUPERVISOR")
public class SupervisorUser extends User {
    
    @Column(name = "occupation")
    private String occupation; // Nghề nghiệp của người thân

    @Column(name = "workplace")
    private String workplace; // Nơi làm việc

    @OneToMany(mappedBy = "supervisorUser", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<ElderSupervisor> elderRelations;
}
