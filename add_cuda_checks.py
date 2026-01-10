#!/usr/bin/env python3
"""
Script to add CUDA error checking to thermal_lbm.cu
"""
import re
import sys

def add_cuda_error_checking(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    i = 0

    # Add error checking macros after includes
    macros_added = False

    while i < len(lines):
        line = lines[i]
        output_lines.append(line)

        # Add macros after #include statements
        if not macros_added and line.startswith('#include') and i + 1 < len(lines) and lines[i + 1].strip() == '':
            # Find the last include
            j = i + 1
            while j < len(lines) and (lines[j].startswith('#include') or lines[j].strip() == ''):
                output_lines.append(lines[j])
                j += 1

            # Add macros
            output_lines.append('\n')
            output_lines.append('// CUDA error checking macros\n')
            output_lines.append('#define CUDA_CHECK(call) do { \\\n')
            output_lines.append('    cudaError_t err = call; \\\n')
            output_lines.append('    if (err != cudaSuccess) { \\\n')
            output_lines.append('        fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \\\n')
            output_lines.append('        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \\\n')
            output_lines.append('    } \\\n')
            output_lines.append('} while(0)\n')
            output_lines.append('\n')
            output_lines.append('#define CUDA_CHECK_KERNEL() do { \\\n')
            output_lines.append('    cudaError_t err = cudaGetLastError(); \\\n')
            output_lines.append('    if (err != cudaSuccess) { \\\n')
            output_lines.append('        fprintf(stderr, "Kernel launch error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \\\n')
            output_lines.append('        throw std::runtime_error(std::string("Kernel error: ") + cudaGetErrorString(err)); \\\n')
            output_lines.append('    } \\\n')
            output_lines.append('} while(0)\n')
            output_lines.append('\n')
            macros_added = True
            i = j
            continue

        # Check for kernel launches (pattern: <<<grid, block>>>)
        if '<<<' in line and '>>>' in line:
            # Add kernel error check on next line after kernel launch
            output_lines.append('    CUDA_CHECK_KERNEL();\n')

        # Check for cudaDeviceSynchronize without error checking
        if 'cudaDeviceSynchronize()' in line and 'CUDA_CHECK' not in line:
            output_lines[-1] = line.replace('cudaDeviceSynchronize()', 'CUDA_CHECK(cudaDeviceSynchronize())')

        # Check for cudaMemcpy without error checking
        if re.match(r'\s*cudaMemcpy\(', line) and 'CUDA_CHECK' not in line:
            output_lines[-1] = line.replace('cudaMemcpy(', 'CUDA_CHECK(cudaMemcpy(')
            # Add closing parenthesis
            if line.strip().endswith(');'):
                output_lines[-1] = output_lines[-1].replace(');', '));')

        # Check for cudaMemset without error checking
        if re.match(r'\s*cudaMemset\(', line) and 'CUDA_CHECK' not in line and 'CUDA_CHECK' not in output_lines[-1]:
            output_lines[-1] = line.replace('cudaMemset(', 'CUDA_CHECK(cudaMemset(')
            # Add closing parenthesis
            if line.strip().endswith(');'):
                output_lines[-1] = output_lines[-1].replace(');', '));')

        # Check for cudaMalloc without error checking
        if re.match(r'\s*cudaMalloc\(', line) and 'CUDA_CHECK' not in line and 'err =' not in line:
            output_lines[-1] = line.replace('cudaMalloc(', 'CUDA_CHECK(cudaMalloc(')
            # Add closing parenthesis
            if line.strip().endswith(');'):
                output_lines[-1] = output_lines[-1].replace(');', '));')

        i += 1

    with open(output_file, 'w') as f:
        f.writelines(output_lines)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)

    add_cuda_error_checking(sys.argv[1], sys.argv[2])
    print(f"Added CUDA error checking to {sys.argv[2]}")
