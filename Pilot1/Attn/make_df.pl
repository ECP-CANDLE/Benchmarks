$EPOCH=200;
$SIZE=240000;
$TYPE='bysize';

while(<>){
	chomp; 
	my ($size, $layer, $epoch, $val)=split; 
	$h{$layer}->{$epoch}->{$size} = $val;
	$epochs{$epoch}++;
	$sizes{$size}++;
}


# layers are rows, epochs are columns
if($TYPE eq 'byepoch') {
	print "\# size = $SIZE\n";
	print ",", join(",", sort {$a <=> $b} keys(%epochs)), "\n";
	foreach my $layer (sort {$a <=> $b} keys(%h)) {
		print $layer;
		foreach $epoch (sort {$a <=> $b} keys(%{$h{$layer}})) {
			foreach $size (sort {$a <=> $b} keys(%{$h{$layer}->{$epoch}})) {
				print ",$h{$layer}->{$epoch}->{$size}" if $size == $SIZE;
			}
		}
		print "\n";
	}
}

# layers are rows, size are columns
if($TYPE eq 'bysize') {
	print "\# epoch = $EPOCH\n";
	print ",", join(",", sort {$a <=> $b} keys(%sizes)), "\n";
	foreach my $layer (sort {$a <=> $b} keys(%h)) {
		print $layer;
		foreach $size (sort {$a <=> $b} keys(%sizes)) {
			print ",", $h{$layer}->{$EPOCH}->{$size};
		}
		print "\n";
	}
}
