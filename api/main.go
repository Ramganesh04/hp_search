package main

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

func getJob(jobsCollection *mongo.Collection) gin.HandlerFunc {
	return func(c *gin.Context) {
		var varHyper Hyperparameters
		if err := c.ShouldBindJSON(&varHyper); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		job := job{
			TimeCreated:     time.Now(),
			TotalParameters: varHyper.Gates * varHyper.NetworkLayers,
			Hyperparameters: varHyper,
			Status:          "Pending",
		}
		ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
		defer cancel()

		result, err := jobsCollection.InsertOne(ctx, job)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"id": result.InsertedID})
	}
}

func updateJob(jobsCollection *mongo.Collection) gin.HandlerFunc {
	return func(c *gin.Context) {
		oid, err := primitive.ObjectIDFromHex(c.Param("id"))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid id"})
			return
		}
		var varHyper Hyperparameters
		if err := c.ShouldBindJSON(&varHyper); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		filter := bson.M{"_id": oid, "status": "Pending"}
		update := bson.M{"$set": bson.M{"hyperparameters": varHyper}}

		ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
		defer cancel()

		result, err := jobsCollection.UpdateOne(ctx, filter, update)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, result)
	}
}
func main() {
	client := connectToMongo()
	db := client.Database(getenv("MONGO_DB", "jobsdb"))
	jobsCollection := db.Collection(getenv("MONGO_COLLECTION", "jobs"))
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		client.Disconnect(ctx)
	}()
	r := gin.Default()
	r.POST("/jobs", getJob(jobsCollection))
	r.PATCH("/jobs/:id", updateJob(jobsCollection))
	r.Run(":8000")
}
