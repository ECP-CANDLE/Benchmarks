
while(<>){
	$c=$c+1; 
	$s=$s+$1 if /(\d+)s /;
}
print "$c\t$s\t", $s/$c, "\n"
