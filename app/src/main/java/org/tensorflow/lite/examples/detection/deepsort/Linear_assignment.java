package org.tensorflow.lite.examples.detection.deepsort;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Linear_assignment {

    static class _HungarianState{

        private final float[][] C;
        private final boolean transposed;
        private final boolean[] row_uncovered;
        private final boolean[] col_uncovered;
        int Z0_r;
        int Z0_c;
        int[][] path;
        int[][] marked;

        private float[][] transpose(float[][] matrix){
            float[][] trans = new float[matrix[0].length][matrix.length];
            for(int i=0;i<matrix[0].length;i++){
                for(int j=0;j<matrix.length;j++)
                    trans[i][j] = matrix[j][i];
            }
            return trans;
        }

        _HungarianState(float[][] cost_matrix){
            boolean transposed = cost_matrix[0].length < cost_matrix.length;
            if(transposed)
                C = transpose(cost_matrix).clone();
            else
                C = cost_matrix.clone();
            this.transposed = transposed;
            int n = C.length;
            int m = C[0].length;
            row_uncovered = new boolean[n];
            Arrays.fill(row_uncovered,true);
            col_uncovered = new boolean[m];
            Arrays.fill(col_uncovered,true);
            Z0_r = 0;
            Z0_c = 0;
            path = new int[n+m][2];
            marked = new int[n][m];
        }

        private void _clear_covers(){
            Arrays.fill(row_uncovered, true);
            Arrays.fill(col_uncovered, true);
        }
    }

    int[][] linear_assignment(float[][] X1){
        float[][] X = new float[X1.length][X1[0].length];
        for(int i=0;i<X1.length;i++)
            System.arraycopy(X1[i], 0, X[i], 0, X1[0].length);
        int[][] indices = _hungarian(X);
        for(int i=0;i<indices.length;i++){
            for(int j=i+1;j<indices.length;j++){
                if(indices[i][0]>indices[j][0]){
                    int[] t = indices[i];
                    indices[i]=indices[j];
                    indices[j]=t;
                }
            }
        }
        if(indices[0].length==2)
            return indices;
        int row = (indices.length * indices[0].length) / 2;
        int[][] new_indices = new int[row][2];
        List<Integer> flat = new ArrayList<>();

        for(int i=0;i<new_indices.length;i++)
            for(int j=0;j<new_indices[0].length;j++)
                flat.add(indices[i][j]);

        int k = 0;
        for(int i=0;i<new_indices.length;i++)
            for(int j=0;j<new_indices[0].length;j++)
                new_indices[i][j] = flat.get(k++);

        return new_indices;
    }

    private int[][] _hungarian(float[][] cost_matrix){

        _HungarianState state = new _HungarianState(cost_matrix);

        if(cost_matrix.length != 0 && cost_matrix[0].length != 0)
            state = _step1(state);

        List<Integer> r = new ArrayList<>();
        List<Integer> c = new ArrayList<>();

        assert state != null;
        for (int i = 0; i < state.marked.length; i++) {
            for (int j = 0; j < state.marked[0].length; j++) {
                if(state.marked[i][j] == 1){
                    r.add(i);
                    c.add(j);
                }
            }
        }

        int[][] results = new int[r.size()][2];

        for (int i = 0; i < results.length; i++)
            results[i] = new int[]{r.get(i), c.get(i)};

        if(state.transposed){
            for (int i = 0; i < results.length; i++) {
                int t = results[i][0];
                results[i][0] = results[i][1];
                results[i][1] = t;
            }
        }

        return results;
    }

    private float min(float[] row){
        float min = row[0];
        for(float r:row) if(r < min) min = r;
        return min;
    }
    private boolean[] np_any(boolean[][] marked){
        boolean[] any = new boolean[marked[0].length];
        for (boolean[] booleans : marked) {
            for (int j = 0; j < booleans.length; j++) {
                any[j] = any[j] || booleans[j];
            }
        }
        return any;
    }
    private boolean np_any(boolean[] a){
        for(boolean x:a){
            if(x)
                return true;
        }
        return false;
    }
    private int[][] np_mul(int[][] a,boolean[] b){
        int[][] C = new int[a.length][a[0].length];
        for(int i =0;i<a.length;i++) {
            if (b[i])
                System.arraycopy(a[i], 0, C[i], 0, a[0].length);
        }
        return C;
    }
    private int[][] np_mul1(int[][] a,boolean[] b){
        int[][] C = new int[a.length][a[0].length];
        for(int i =0;i<a.length;i++) {
            for (int j = 0; j < a[0].length; j++)
                if (b[j]) C[i][j] = a[i][j];
        }
        return C;
    }
    private int np_argmax(int[][] matrix){
        int max = matrix[0][0];
        int argmax = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if(max < matrix[i][j]) {
                    max = matrix[i][j];
                    argmax = i * matrix[0].length + j;
                }
            }
        }
        return argmax;
    }

    private _HungarianState _step1(_HungarianState state) {
        float min;
        for (int i = 0; i < state.C.length; i++) {
            min = min(state.C[i]);
            for (int j = 0; j < state.C[i].length; j++) {
                state.C[i][j] -= min;
            }
        }
        for (int i = 0; i < state.C.length; i++) {
            for (int j = 0; j < state.C[i].length; j++) {
                if (state.C[i][j] == 0 &&
                        state.col_uncovered[j] &&
                        state.row_uncovered[i]) {
                    state.marked[i][j] = 1;
                    state.col_uncovered[j] = false;
                    state.row_uncovered[i] = false;
                }
            }
        }
        state._clear_covers();
        return _step3(state);
    }

    private _HungarianState _step3(_HungarianState state){
        boolean[][] marked = new boolean[state.marked.length][state.marked[0].length];
        for(int i=0;i<marked.length;i++)
            for(int j=0;j<marked[0].length;j++)
                marked[i][j] = state.marked[i][j] == 1;
        boolean[] any_marked = np_any(marked);
        for (int i = 0; i < any_marked.length; i++)
            if(any_marked[i])
                state.col_uncovered[i]=false;
        int sum = 0;
        for (boolean[] booleans : marked)
            for (boolean aBoolean : booleans)
                if (aBoolean) sum += 1;
        if(sum < state.C.length)
            return _step4(state);
        return state;
    }

    private _HungarianState _step4(_HungarianState state){
        int[][] C = new int[state.C.length][state.C[0].length];
        for(int i=0;i<state.C.length;i++)
            for(int j=0;j<state.C[0].length;j++)
                if (state.C[i][j] == 0) C[i][j] = 1;
        int[][] covered_C = np_mul(C,state.row_uncovered);
        covered_C = np_mul1(covered_C,state.col_uncovered);
        int m = state.C[0].length;
        int r,c;
        while (true) {
            r = np_argmax(covered_C) / m;
            c = np_argmax(covered_C) % m;
            if (covered_C[r][c] == 0)
                return _step6(state);
            else{
                state.marked[r][c] = 2;
                int star_col = 0,f=0;
                while(star_col < state.marked[r].length){
                    if(state.marked[r][star_col] == 1){
                        f = 1;
                        break;
                    }
                    star_col++;
                }
                if(f == 0) star_col = 0;
                if(!(state.marked[r][star_col] == 1)){
                    state.Z0_r = r;
                    state.Z0_c = c;
                    return _step5(state);
                }else {
                    c = star_col;
                    state.row_uncovered[r] = false;
                    state.col_uncovered[c] = true;
                    for (int i = 0; i < covered_C.length; i++)
                        covered_C[i][c] = C[i][c] * (state.row_uncovered[i] ? 1 : 0);
                    Arrays.fill(covered_C[r],0);
                }
            }
        }
    }

    private _HungarianState _step5(_HungarianState state){
        int count = 0;
        int[][] path = state.path;
        path[count][0] = state.Z0_r;
        path[count][1] = state.Z0_c;

        while(true){
            //row
            float[] marked = new float[state.marked.length];
            for(int i=0;i<state.marked.length;i++)
                marked[i] = state.marked[i][path[count][1]];
            int row = 0;
            for(int i=0;i<marked.length;i++){
                if(marked[i]==1) {
                    row = i;
                    break;
                }
            }
            if(!(state.marked[row][path[count][1]]==1))
                break;
            else{
                count += 1;
                path[count][0] = row;
                path[count][1] = path[count-1][1];
            }
            //col
            int col = 0;
            for(int i=0;i<state.marked[0].length;i++){
                if(state.marked[path[count][0]][i]==2){
                    col = i;
                    break;
                }
            }
            if(state.marked[row][col] != 2)
                col = -1;
            count += 1;
            path[count][0] = path[count-1][0];
            path[count][1] = col;
        }
        for (int i = 0; i < count+1; i++)
            if(state.marked[path[i][0]][path[i][1]] == 1)
                state.marked[path[i][0]][path[i][1]] = 0;
            else
                state.marked[path[i][0]][path[i][1]] = 1;

        state._clear_covers();
        for (int i = 0; i < state.marked.length; i++)
            for (int j = 0; j < state.marked[0].length; j++)
                if(state.marked[i][j] == 2)
                    state.marked[i][j] = 0;
        return _step3(state);
    }

    private _HungarianState _step6(_HungarianState state){
        if(np_any(state.col_uncovered) && np_any(state.row_uncovered)){
            //float[][] C = new float[state.C.length][state.C[0].length];
            float[] minVal1 = new float[state.C[0].length];
            Arrays.fill(minVal1,100);
            //System.arraycopy(state.C[0], 0, minVal1, 0, minVal1.length);
            for(int i=0;i<state.C.length;i++) {
                if (state.row_uncovered[i]) {
                    for (int j = 0; j < state.C[i].length; j++) {
                        if (state.C[i][j] < minVal1[j]) {
                            minVal1[j] = state.C[i][j];
                        }
                    }
                }
            }
            float minVal = 100;
            for (int i = 0; i < minVal1.length; i++) {
                if(state.col_uncovered[i]){
                    minVal = minVal1[i];
                    break;
                }
            }
            for (int i = 0; i < minVal1.length; i++)
                if(state.col_uncovered[i] && minVal1[i] < minVal)
                    minVal = minVal1[i];
            for (int i = 0; i < state.C.length; i++)
                if(!state.row_uncovered[i])
                    for (int j = 0; j < state.C[0].length; j++)
                        state.C[i][j] += minVal;
            for(int i=0;i<state.C.length;i++)
                for(int j=0;j<state.C[0].length;j++)
                    if(state.col_uncovered[j])
                        state.C[i][j] -= minVal;
        }
        return _step4(state);
    }
}