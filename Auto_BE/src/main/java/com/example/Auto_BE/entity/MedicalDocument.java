package com.example.Auto_BE.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.util.List;
import java.util.Set;

@Entity
@Table(name = "medical_documents")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Accessors(chain = true)
public class MedicalDocument extends BaseEntity {
    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "description", nullable = false)
    private String description;

    @OneToMany(mappedBy = "medicalDocument")
    private List<MedicalDocumentFile> files; // Danh sách các tệp đính kèm liên quan đến tài liệu y tế

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user; // Người dùng sở hữu tài liệu y tế này
}