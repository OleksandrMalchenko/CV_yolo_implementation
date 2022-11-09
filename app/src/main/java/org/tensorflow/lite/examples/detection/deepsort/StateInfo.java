package org.tensorflow.lite.examples.detection.deepsort;

import java.util.ArrayList;
import java.util.List;

public class StateInfo {

    private int id;
    private float[] mbbox;
    private float[] sbbox;
    private ArrayList<float[]> mfeature;
    private ArrayList<float[]> sfeature;
    private int timestamp;
    private int missing_ts;
    private float[] velocity;
    private boolean stationary_flag;
    private List<Integer> occ_list;
    private long update_frame;
    private int ROI;
    private boolean entry1;
    private boolean entry2;
    private boolean exit1;
    private boolean exit2;

    public boolean isExit1() {
        return exit1;
    }

    public void setExit1(boolean exit1) {
        this.exit1 = exit1;
    }

    public boolean isExit2() {
        return exit2;
    }

    public void setExit2(boolean exit2) {
        this.exit2 = exit2;
    }

    private boolean random;

    public boolean isEntry1() {
        return entry1;
    }

    public void setEntry1(boolean entry1) {
        this.entry1 = entry1;
    }

    public boolean isEntry2() {
        return entry2;
    }

    public void setEntry2(boolean entry2) {
        this.entry2 = entry2;
    }
    private final long first_ts;

    public long getFirst_ts() { return first_ts; }

    public boolean isRandom() { return random; }

    public void setRandom(boolean rand) { this.random = rand; }

    public int getROI() { return ROI; }

    public void setROI(int ROI_val) {
        this.ROI = ROI_val;
    }

    float[] getSbbox() {
        return sbbox;
    }

    ArrayList<float[]> getMfeature() {
        return mfeature;
    }

    ArrayList<float[]> getSfeature() {
        return sfeature;
    }

    int getTimestamp() {
        return timestamp;
    }

    int getMissing_ts() {
        return missing_ts;
    }

    float[] getVelocity() {
        return velocity;
    }

    public boolean isStationary_flag() {
        return stationary_flag;
    }

    List<Integer> getOcc_list() {
        return occ_list;
    }

    public float[] getMbbox() {
        return mbbox;
    }

    void setId(int id) {
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public long getUpdate_frame() {
        return update_frame;
    }

    void setUpdate_frame(long update_frame) {
        this.update_frame = update_frame;
    }

    void setMbbox(float[] mbbox) {
        this.mbbox = mbbox;
    }

    void setSbbox(float[] sbbox) {
        this.sbbox = sbbox;
    }

    void setMfeature(ArrayList<float[]> mfeature) {
        this.mfeature = mfeature;
    }

    void setSfeature(ArrayList<float[]> sfeature) {
        this.sfeature = sfeature;
    }

    void setTimestamp(int timestamp) {
        this.timestamp = timestamp;
    }

    void setMissing_ts(int missing_ts) {
        this.missing_ts = missing_ts;
    }

    void setVelocity(float[] velocity) {
        this.velocity = velocity;
    }

    void setStationary_flag(boolean stationary_flag) {
        this.stationary_flag = stationary_flag;
    }

    void setOcc_list(List<Integer> occ_list) {
        this.occ_list = occ_list;
    }

    StateInfo(int id1,float[] mbbox1,float[] sbbox1,ArrayList<float[]> mfeature1,ArrayList<float[]> sfeature1,int timestamp1,int missing_ts1,float[] velocity1,boolean stationary_flag1,List<Integer> occ_list1,long ts){
        id = id1;
        mbbox = mbbox1;
        sbbox = sbbox1;
        mfeature = mfeature1;
        sfeature = sfeature1;
        timestamp = timestamp1;
        missing_ts = missing_ts1;
        velocity = velocity1;
        stationary_flag = stationary_flag1;
        occ_list = occ_list1;
        update_frame = ts;
        random = true;
        first_ts = ts;
        entry1 = false;
        entry2 = false;
        exit1 = false;
        exit2 = false;
    }
}
