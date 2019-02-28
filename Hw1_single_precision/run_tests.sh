make

k=2
for i in {1..10}
do
	b=$(echo $((2**i));)
	./main -f -n 1000 -m 1000 -b $b 
done

for i in {1..10}
do
	b=$(echo $((2**i));)
	./main -f -n 5000 -m 5000 -b $b
done

for i in {1..10}
do
	b=$(echo $((2**i));)
	./main -f -n 10000 -m 10000 -b $b
done

for i in {1..10}
do
	b=$(echo $((2**i));)
	./main -f -n 30000 -m 30000 -b $b 
done


