.PHONY = exc exp clean

exc:
	mpicxx -o int_calc main.cpp

exp:
	@echo "Start experiment: integral step is 1000000"
	mpicxx -o int_calc main.cpp
	chmod 755 exp.sh
	./exp.sh
clean:
	@echo "Cleaning up..."
	@rm -f int_calc *.txt

