# -*- coding: utf-8 -*-                                                                                     
from pymol.cgo import *                                                                                     
from pymol import cmd                                                                                       
from random import randint                                                                                  


def draw_grid(minX, minY, minZ, maxX, maxY, maxZ, linewidth, r, g, b):
        minX, minY, minZ = float(minX), float(minY), float(minZ)
        maxX, maxY, maxZ = float(maxX), float(maxY), float(maxZ)
        boundingBox = [
                LINEWIDTH, float(linewidth),

                BEGIN, LINES,
                COLOR, float(r), float(g), float(b),

                VERTEX, minX, minY, minZ,       #1
                VERTEX, minX, minY, maxZ,       #2

                VERTEX, minX, maxY, minZ,       #3
                VERTEX, minX, maxY, maxZ,       #4

                VERTEX, maxX, minY, minZ,       #5
                VERTEX, maxX, minY, maxZ,       #6

                VERTEX, maxX, maxY, minZ,       #7
                VERTEX, maxX, maxY, maxZ,       #8


                VERTEX, minX, minY, minZ,       #1
                VERTEX, maxX, minY, minZ,       #5

                VERTEX, minX, maxY, minZ,       #3
                VERTEX, maxX, maxY, minZ,       #7

                VERTEX, minX, maxY, maxZ,       #4
                VERTEX, maxX, maxY, maxZ,       #8

                VERTEX, minX, minY, maxZ,       #2
                VERTEX, maxX, minY, maxZ,       #6


                VERTEX, minX, minY, minZ,       #1
                VERTEX, minX, maxY, minZ,       #3

                VERTEX, maxX, minY, minZ,       #5
                VERTEX, maxX, maxY, minZ,       #7

                VERTEX, minX, minY, maxZ,       #2
                VERTEX, minX, maxY, maxZ,       #4

                VERTEX, maxX, minY, maxZ,       #6
                VERTEX, maxX, maxY, maxZ,       #8

                END
        ]

        boxName = "box_" + str(randint(0,10000))
        while boxName in cmd.get_names():
                boxName = "box_" + str(randint(0,10000))

        cmd.load_cgo(boundingBox,boxName)
        return boxName

      


def draw_box(minX=0.0, minY=0.0, minZ=0.0, maxX=1.0, maxY=1.0, maxZ=1.0, nlen=1.5, padding=0.0, linewidth=2.0, r=1.0, g=1.0, b=1.0):     
        """                                                                  
        DESCRIPTION                                                          
                Given selection, draw the bounding box around it.          

        USAGE:
                drawBoundingBox [selection, [padding, [linewidth, [r, [g, b]]]]]

        PARAMETERS:
   
                padding,                defaults to 0

                linewidth,              width of box lines
                                        defaults to 2.0

                r,                      red color component, valid range is [0.0, 1.0]
                                        defaults to 1.0                               

                g,                      green color component, valid range is [0.0, 1.0]
                                        defaults to 1.0                                 

                b,                      blue color component, valid range is [0.0, 1.0]
                                        defaults to 1.0                                

        RETURNS
                string, the name of the CGO box

        NOTES
                * This function creates a randomly named CGO box that minimally spans the protein. The
                user can specify the width of the lines, the padding and also the color.                            
        """                                                                                                    
        minX, minY, minZ = float(minX), float(minY), float(minZ)
        maxX, maxY, maxZ = float(maxX), float(maxY), float(maxZ)
        nlen = float(nlen)

        print( "Box dimensions (%.2f, %.2f, %.2f)" % (maxX-minX, maxY-minY, maxZ-minZ) )

        minX = minX - float(padding)
        minY = minY - float(padding)
        minZ = minZ - float(padding)
        maxX = maxX + float(padding)
        maxY = maxY + float(padding)
        maxZ = maxZ + float(padding)

        if padding != 0:
                 print( "Box dimensions + padding (%.2f, %.2f, %.2f)" % (maxX-minX, maxY-minY, maxZ-minZ) )

        nX = int((maxX - minX)/nlen) if (maxX - minX) > nlen else 1
        dX = (maxX - minX) / int((maxX -minX)/nlen) if  (maxX - minX) > nlen else (maxX - minX)
        nY = int((maxY - minY)/nlen) if (maxY - minY) > nlen else 1
        dY = (maxY - minY) / int((maxY - minY)/nlen) if (maxY - minY) > nlen else (maxY - minY)
        nZ = int((maxZ - minZ)/nlen) if (maxZ - minZ) > nlen else 1
        dZ = (maxZ - minZ) / int((maxZ - minZ)/nlen) if (maxZ - minZ) > nlen else (maxZ - minZ)
        for xi in range(nX):
            for yi in range(nY):
                for zi in range(nZ):
                    gminX, gminY, gminZ = minX + xi * dX, minY + yi * dY, minZ + zi * dZ
                    gmaxX, gmaxY, gmaxZ = minX + (xi+1) * dX, minY + (yi+1) * dY, minZ + (zi + 1) * dZ
                    draw_grid(minX=gminX, minY=gminY, minZ=gminZ, maxX=gmaxX, maxY=gmaxY, maxZ=gmaxZ, linewidth=linewidth, r=r, g=g, b=b)
        return draw_grid(minX=minX, minY=minY, minZ=minZ, maxX=maxX, maxY=maxY, maxZ=maxZ, linewidth=linewidth, r=r, g=g, b=b)




def draw_signature(minX=0.0, minY=0.0, minZ=0.0, 
                   maxX=5.0, maxY=5.0, maxZ=5.0, 
                   padding=2.0, 
                   sig_partition=(5, 5, 5), mark_grid=(1, 2, 1),linewidth=2.0, 
                   r=1.0, g=1.0, b=1.0):
        minX, minY, minZ = float(minX), float(minY), float(minZ)
        maxX, maxY, maxZ = float(maxX), float(maxY), float(maxZ)
        nX = int(sig_partition[0])
        nY = int(sig_partition[1])
        nZ = int(sig_partition[2])
        
        minX = minX - float(padding)
        minY = minY - float(padding)
        minZ = minZ - float(padding)
        maxX = maxX + float(padding)
        maxY = maxY + float(padding)
        maxZ = maxZ + float(padding)
        
        dX = (maxX - minX) / nX
        dY = (maxY - minY) / nY
        dZ = (maxZ - minZ) / nZ

        gX = int(mark_grid[0]) - 1
        gY = int(mark_grid[1]) - 1
        gZ = int(mark_grid[2]) - 1
        gminX, gminY, gminZ = minX + gX * dX, minY + gY * dY,  minZ + gZ * dZ
        gmaxX, gmaxY, gmaxZ = minX + (gX+2) * dX, minY + (gY + 2) * dY, minZ + (gZ + 2) * dZ
        return draw_grid(minX=gminX, minY=gminY, minZ=gminZ, maxX=gmaxX, maxY=gmaxY, maxZ=gmaxZ, linewidth=linewidth, r=r, g=g, b=b)

cmd.extend("drawSignatureBox", draw_signature)
cmd.extend("drawBoundingBox", draw_box)
