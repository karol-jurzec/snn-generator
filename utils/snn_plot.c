#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>
#include <stdlib.h>

void plot_single_neuron(const char *filename, const char *output_png) {
    FILE *gnuplot = popen("gnuplot -persistent", "w");
    if (gnuplot == NULL) {
        perror("Error opening gnuplot");
        exit(EXIT_FAILURE);
    }

    // Gnuplot commands to save the plot as a PNG
    fprintf(gnuplot, "set terminal pngcairo enhanced font 'Arial,10' size 800,600\n"); // Set PNG terminal
    fprintf(gnuplot, "set output '%s'\n", output_png); // Specify output file
    fprintf(gnuplot, "set title 'Neuron Membrane Potential'\n");
    fprintf(gnuplot, "set xlabel 'Time Step'\n");
    fprintf(gnuplot, "set ylabel 'Membrane Potential (mV)'\n");
    fprintf(gnuplot, "set grid\n");
    fprintf(gnuplot, "plot '%s' using 1:2 with lines title 'Membrane Potential'\n", filename);

    // Close the gnuplot pipe
    pclose(gnuplot);
}

