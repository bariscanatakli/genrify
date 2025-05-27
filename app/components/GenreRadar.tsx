"use client";

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface GenreRadarProps {
  genreProbabilities: Record<string, number>;
  width?: number;
  height?: number;
}

const GenreRadar: React.FC<GenreRadarProps> = ({ 
  genreProbabilities, 
  width = 300, 
  height = 300 
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || Object.keys(genreProbabilities).length === 0) return;

    // Clear previous visualization
    d3.select(svgRef.current).selectAll("*").remove();

    // Extract data
    const genres = Object.keys(genreProbabilities);
    const values = Object.values(genreProbabilities);
    
    const data = genres.map((genre, i) => ({
      genre,
      value: values[i]
    }));
    
    const svg = d3.select(svgRef.current);
    const margin = { top: 30, right: 30, bottom: 30, left: 30 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const radius = Math.min(chartWidth, chartHeight) / 2;
    
    const g = svg.append("g")
      .attr("transform", `translate(${width/2}, ${height/2})`);
    
    // Create scales
    const angleScale = d3.scalePoint<string>()
      .domain(genres)
      .range([0, 2 * Math.PI]);
    
    const radiusScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, radius]);
    
    // Draw background circles
    const circles = [0.2, 0.4, 0.6, 0.8];
    circles.forEach(c => {
      g.append("circle")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("r", radiusScale(c))
        .attr("fill", "none")
        .attr("stroke", "currentColor")
        .attr("stroke-dasharray", "2,2")
        .attr("stroke-width", 0.5)
        .attr("opacity", 0.3);
    });
    
    // Draw axis lines
    genres.forEach(genre => {
      const angle = angleScale(genre) || 0;
      const x = radius * Math.sin(angle);
      const y = -radius * Math.cos(angle);
      
      g.append("line")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", x)
        .attr("y2", y)
        .attr("stroke", "currentColor")
        .attr("stroke-width", 0.5)
        .attr("opacity", 0.3);
        
      // Add labels
      g.append("text")
        .attr("x", 1.1 * x)
        .attr("y", 1.1 * y)
        .attr("text-anchor", angle > Math.PI ? "end" : "start")
        .attr("dominant-baseline", "middle")
        .attr("font-size", "10px")
        .attr("opacity", 0.8)
        .text(genre);
    });
    
    // Create line generator
    const lineGenerator = d3.lineRadial<{genre: string, value: number}>()
      .angle(d => angleScale(d.genre) || 0)
      .radius(d => radiusScale(d.value));
    
    // Draw the shape
    g.append("path")
      .datum(data)
      .attr("d", lineGenerator as any)
      .attr("fill", "rgba(59, 130, 246, 0.3)")
      .attr("stroke", "rgb(59, 130, 246)")
      .attr("stroke-width", 2);
      
    // Add dots
    g.selectAll(".dots")
      .data(data)
      .join("circle")
      .attr("cx", d => radiusScale(d.value) * Math.sin(angleScale(d.genre) || 0))
      .attr("cy", d => -radiusScale(d.value) * Math.cos(angleScale(d.genre) || 0))
      .attr("r", 4)
      .attr("fill", "rgb(59, 130, 246)");
      
  }, [genreProbabilities, width, height]);

  return (
    <svg ref={svgRef} width={width} height={height}></svg>
  );
};

export default GenreRadar;
