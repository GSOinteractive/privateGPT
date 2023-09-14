'use client';
import styles from '@styles/main.css';
import { Row, Container, Col, Stack } from 'react-bootstrap';
import ConfigSideNav from '@components/ConfigSideNav';
import MainContainer from '@components/MainContainer';
import { ToastContainer, toast } from 'react-toastify';

export default function Home() {
  return (
    <>
      <Row className=" p-3 vh-100 overflow-hidden  g-0">
        <Col lg={12} xs={12} className="main-chat-col">
          <MainContainer />
        </Col>
        <ToastContainer />
      </Row>
    </>
  );
}
