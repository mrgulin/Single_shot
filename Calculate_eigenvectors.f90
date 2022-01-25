program Calculate_eigenvectors
	implicit none 
	integer :: i, ix, size_matrix
	REAL(KIND=8), DIMENSION (:,:), ALLOCATABLE :: A, ei_vec
	REAL(KIND=8), DIMENSION (:), ALLOCATABLE :: ei_val
	OPEN(unit=99, file='temp_size.dat')
	read(99, *) size_matrix
	close(99)
	ALLOCATE(A(size_matrix, size_matrix), ei_vec(size_matrix, size_matrix))
	ALLOCATE(ei_val(size_matrix))
		
	A = 0
	OPEN(unit=100, file='temp_matrix.dat')
	
	
	
	print *, size_matrix
	
	!do ix = 1,size_matrix
	read(100,*) A
	CLOSE(100)
	!end do
		PRINT *, "A" !debug
	do, i=1, size_matrix
		print '(20f12.6)', A(i,:)
	enddo
	
	CALL DIAGMAT(size_matrix,A,ei_val,ei_vec)
	
	OPEN(unit=101, file='temp_ei_vec.dat')
	OPEN(unit=102, file='temp_ei_val.dat')
	do, i=1, size_matrix
		WRITE (101,*) ei_vec(i,:) !'(20f12.6)'
	enddo 
	WRITE (102, *) ei_val
	CLOSE(101)
	CLOSE(102)
	
	

end program Calculate_eigenvectors