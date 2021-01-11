var inputDir;
var outputDir;
var formatImage = "jpg";

macro enleverFond_PdT{

	Dialog.create("Extraction_Fond");
	Dialog.addString("Type image:", "jpg");
	Dialog.show;
	formatImage = Dialog.getString;

	inputDir = getDirectory("Choisir le répertoire où sont stockées les photos");
	outputDir = getDirectory("Choisir un répertoire de destination des résultats");
	inputFiles = getFileList(inputDir);
	for (i = 0 ; i < inputFiles.length ; i++){
		if (endsWith(inputFiles[i], formatImage)){
			process(inputFiles[i]);
	}
}	
}

function process(imageFileName){

	// open image
	open(inputDir + File.separator + imageFileName);
	replace(imageFileName, ".JPG", "");
	imageOriginale = getTitle();
		
	// 1
	run("Duplicate...", " ");
	imageOriginale1 = getTitle();
	run("Apply saved SIOX segmentator", "browse=./segmentator.siox");
	run("Fill Holes");
	imageOriginale2 = getTitle();
	run("Duplicate...", " ");
	run("Outline");
	run("Options...", "iterations=3 count=1 do=Dilate");
	run("Invert");
	run("Magenta");
	imageContour = getTitle();
	imageCalculator("OR create", imageOriginale1, imageContour);
	saveAs("jpg", outputDir + File.separator + imageOriginale + "_Cont");
	close();
    
	// 2
	imageCalculator("OR create", imageOriginale, imageOriginale2);
	saveAs("jpg", outputDir + File.separator + imageOriginale);
	
	run("Close All");
	}
	waitForUser("Traitement termine !!");
